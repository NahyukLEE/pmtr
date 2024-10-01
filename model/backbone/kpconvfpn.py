import torch
import torch.nn as nn

from model.backbone.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample
from model.pmt import PMTBlock

class KPConvFPN_Flex(nn.Module):
    def __init__(self, input_dim, output_dim, channels, fine_matcher):
        super(KPConvFPN_Flex, self).__init__()

        kernel_size = 15
        init_radius = 0.125
        init_sigma = 0.1
        group_norm = 32

        self.fine_matcher = fine_matcher
        assert len(channels) - 1 == len(output_dim)

        encoders, decoders = [], []
        for idx, layer_channels in enumerate(channels):

            blocks = []
            if idx == 0:
                blocks.append(ConvBlock(input_dim, layer_channels[0], kernel_size, init_radius, init_sigma, group_norm))
                inch = layer_channels[0]
                for outch in layer_channels[1:]:
                    blocks.append(ResidualBlock(inch, outch, kernel_size, init_radius, init_sigma, group_norm))
                    inch = outch
            else:
                blocks.append(ResidualBlock(inch, layer_channels[0], kernel_size, init_radius, init_sigma, group_norm, strided=True))
                init_radius *= 2
                init_sigma *= 2
                inch = layer_channels[0]
                for outch in layer_channels[1:]:
                    blocks.append(ResidualBlock(inch, outch, kernel_size, init_radius, init_sigma, group_norm))
                    inch = outch
            blocks = nn.ModuleList(blocks)
            encoders.append(blocks)

        if self.fine_matcher == 'none':
            inch = channels[-1][-1] + channels[-2][-1]
            for idx, outch in enumerate(output_dim):
                if idx == len(output_dim) - 1:
                    decoders.append(LastUnaryBlock(inch, outch))
                else:
                    decoders.append(UnaryBlock(inch, outch, group_norm))
                    inch = outch + channels[-3 - idx][-1]
            self.encoders = nn.ModuleList(encoders)
            self.decoders = nn.ModuleList(decoders)

        elif self.fine_matcher == 'pmt':
            from model.pmt import PMTBlock
            chdim = {'0': [[64, 16, 4], [4, 16, 64]], '1': [[64, 16, 4], [2, 8, 32]]}

            inch = channels[-1][-1] + channels[-2][-1]
            
            self.pmt = []
            for idx, outch in enumerate(output_dim):
                if idx == len(output_dim) - 1:
                    decoders.append(LastUnaryBlock(inch, outch))
                else:
                    decoders.append(UnaryBlock(inch, outch, group_norm))
                    inch = outch + channels[-3 - idx][-1]
                crdim, ftdim = chdim[str(idx)]
                self.pmt.append(PMTBlock(crdim, ftdim, [4, 4], 'dist', 0.1, neighbor=True))
                
            self.pmt = nn.ModuleList(self.pmt)
            self.encoders = nn.ModuleList(encoders)
            self.decoders = nn.ModuleList(decoders)

    def forward(self, feats, data_dict):
        enc_feats_list = []
        feats_list = []
        points_list = [x.squeeze(0) for x in data_dict['points_ext_t'][list(data_dict['points_ext_t'].keys())[0]]]
        neighbors_list = [x.squeeze(0) for x in data_dict['neighbors_ext_t'][list(data_dict['neighbors_ext_t'].keys())[0]]]
        subsampling_list = [x.squeeze(0) for x in data_dict['subsampling_ext_t'][list(data_dict['subsampling_ext_t'].keys())[0]]]
        upsampling_list = [x.squeeze(0) for x in data_dict['upsampling_ext_t'][list(data_dict['upsampling_ext_t'].keys())[0]]]

        in_feat = feats
        for encoder in self.encoders[0]:
            in_feat = encoder(in_feat, points_list[0], points_list[0], neighbors_list[0])
        enc_feats_list.append(in_feat)
        for idx, encoder in enumerate(self.encoders[1:]):
            for jdx, block in enumerate(encoder):
                if jdx == 0:
                    in_feat = block(in_feat, points_list[idx + 1], points_list[idx], subsampling_list[idx])
                else:
                    in_feat = block(in_feat, points_list[idx + 1], points_list[idx + 1], neighbors_list[idx + 1])
            enc_feats_list.append(in_feat)
        enc_feats_list.reverse()
        feats_list.append(enc_feats_list[0])
        latent_feat = enc_feats_list[0]
        nlayers = len(enc_feats_list)

        if self.fine_matcher == 'none':
            for idx, (merge_feat, decoder) in enumerate(zip(enc_feats_list[1:], self.decoders)):
                latent_feat = nearest_upsample(latent_feat, upsampling_list[nlayers - idx - 2])
                latent_feat = torch.cat([latent_feat, merge_feat], dim=1)
                latent_feat = decoder(latent_feat)
                feats_list.append(latent_feat)

        elif self.fine_matcher == 'pmt':
            for idx, (merge_feat, decoder, pmt) in enumerate(zip(enc_feats_list[1:], self.decoders, self.pmt)):
                latent_feat = nearest_upsample(latent_feat, upsampling_list[nlayers - idx - 2])
                latent_feat = torch.cat([latent_feat, merge_feat], dim=1)
                latent_feat = decoder(latent_feat)

                target_idx = len(points_list) - idx - 2
                latent_feat = pmt.forward_with_neighbor(latent_feat, points_list[target_idx], neighbors_list[target_idx], data_dict['lengths_ext_t'][list(data_dict['lengths_ext_t'].keys())[0]][target_idx][0])

                feats_list.append(latent_feat)

        feats_list.reverse()
        return feats_list
