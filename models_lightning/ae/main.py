import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

from models.ae.ConvLSTMCell import ConvLSTMCell

class deepSWE(pl.LightningModule):
    def __init__(self, nf, in_chan, out_chan, future_frames=1, image_size=256):
        super(deepSWE, self).__init__()

        self.nf = nf
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.future_frames = future_frames
        self.image_size = image_size

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(5, 5),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(5, 5),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=out_chan,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

    # ------------------------------------------------

    def sse_loss(self, input, target):
        return torch.sum((target - input)**2)

    def mse_loss(self, input, target):
        return F.mse_loss(input, target, reduction='sum') # reduction='sum'

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    '''
    def training_epoch_end(self, outputs):
        if (self.current_epoch == 0): # save computational graph
            sampleImg = torch.rand((1, 4, 4, self.image_size, self.image_size))
            self.logger.experiment.add_graph(deepSWE(self.nf, self.in_chan, self.out_chan), sampleImg)
    '''

    def training_step(self, train_batch, batch_idx):

        x, y = train_batch

        center = int(x.shape[3]/3)
        start = center
        end = 2*center

        logits = self.forward(x, self.future_frames)
        logits = logits.permute(0, 2, 1, 3, 4)

        loss = self.sse_loss(logits[:,:,:,start:end, start:end], y[:,:,:,start:end, start:end])
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        center = int(x.shape[3] / 3)
        start = center
        end = 2 * center

        logits = self.forward(x, self.future_frames)
        logits = logits.permute(0, 2, 1, 3, 4)

        loss = self.sse_loss(logits[:, :, :, start:end, start:end], y[:, :, :, start:end, start:end])
        self.log('val_loss', loss)


    # ------------------------------------------------

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.ReLU()(outputs)

        return outputs

    # ------------------------------------------------

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
