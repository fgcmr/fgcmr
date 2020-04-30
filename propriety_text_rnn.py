import argparse
import torch
import os
from torch import nn
import torch.nn.functional as F
from rnndataset import CubTextDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=500, emb_dim=100, emb_vectors=None,
                 emb_dropout=0.3,
                 lstm_dim=1024, lstm_n_layer=2, lstm_dropout=0.3,
                 bidirectional=True, lstm_combine='add',
                 n_linear=2, linear_dropout=0.5, n_classes=200,
                 crit=nn.CrossEntropyLoss()):
        super().__init__()
        vocab_size, emb_dim = emb_vectors.shape
        n_dirs = bidirectional + 1
        lstm_dir_dim = lstm_dim // n_dirs if lstm_combine == 'concat' else lstm_dim

        self.lstm_n_layer = lstm_n_layer
        self.n_dirs = n_dirs
        self.lstm_dir_dim = lstm_dir_dim
        self.lstm_combine = lstm_combine

        self.embedding_layer = nn.Embedding(*emb_vectors.shape)
        self.embedding_layer.from_pretrained(emb_vectors, padding_idx=1)
        # pad=1 in torchtext; embedding weights trainable
        self.embedding_dropout = nn.Dropout(p=emb_dropout)

        self.lstm = nn.LSTM(emb_dim, lstm_dir_dim,
                            num_layers=lstm_n_layer,
                            bidirectional=bidirectional,
                            batch_first=True)
        if lstm_n_layer > 1: self.lstm.dropout = lstm_dropout
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

        self.att_w = nn.Parameter(torch.randn(1, lstm_dim, 1))
        self.linear_layers = [nn.Linear(lstm_dim, lstm_dim) for _ in
                              range(n_linear - 1)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.linear_dropout = nn.Dropout(p=linear_dropout)

        self.label = nn.Linear(lstm_dim, n_classes)
        self.crit = crit

        self.opts = {
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'emb_dropout': emb_dropout,
            'emb_vectors': emb_vectors,
            'lstm_dim': lstm_dim,
            'lstm_n_layer': lstm_n_layer,
            'lstm_dropout': lstm_dropout,
            'lstm_combine': lstm_combine,
            'n_linear': n_linear,
            'linear_dropout': linear_dropout,
            'n_classes': n_classes,
            'crit': crit,
        }

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(
            2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(
            2)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size,
                               self.lstm_dir_dim)[-1]
        final_h = final_h.permute(1, 0, 2)
        final_h = final_h.sum(dim=1)  # (batch_size, 1, self.half_dim)

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        if self.lstm_combine == 'add':
            lstm_output = lstm_output.view(batch_size, seq_len, 2,
                                           self.lstm_dir_dim)#[64, 448, 2, 256]
            lstm_output = lstm_output.sum(dim=2)#[64, 448, 256]
            # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input):
        batch_size, seq_len, *_ = input.shape
        # print('input',input.size())#[4,448]
        inp = self.embedding_layer(input)
        # print('embedding_layer', inp.size())#[4,448,224]即[batch_size, seq_len, embed_size]
        inp = self.embedding_dropout(inp)

        lstm_output, (final_h, final_c) = self.lstm(inp)
        # print('lstm_output',lstm_output.size())#[4,448,512]
        # outputs = []
        # for i in range(seq_len):
        #     cur_emb = inp[i:i + 1]  # .view(1, inp.size(1), inp.size(2))
        #
        #     o, hidden = self.lstm(cur_emb) if i == 0 else self.lstm(cur_emb, hidden)
        #     import pdb;pdb.set_trace()
        #     outputs += [o.unsqueeze(0)]
        #
        # outputs = torch.cat(outputs, dim=0)

        lstm_output = self.lstm_dropout(lstm_output)

        attn_output = self.re_attention(lstm_output, final_h, input)
        output = self.linear_dropout(attn_output)
        # print('attn_output', attn_output.size())#[4,256]
        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)
        # print('output',output.size())#[4,256]
        # print('output', output)
        logits = self.label(output)
    #    print('logits',logits.size())#[4,200]
        return logits

    def forward_normal_attention(self):
        batch_size = len(input)

        inp = self.embedding_layer(input)
        inp = self.embedding_dropout(inp)
        lstm_output, (final_h, final_c) = self.lstm(inp)
        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size,
                               self.lstm_dim // self.n_dirs)[-1]
        final_h = final_h.permute(1, 0,
                                  2)  # (batch_size, 2, self.lstm_dim // self.n_dirs)
        final_h = final_h.contiguous().view(batch_size, self.lstm_dim)

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(lstm_output, final_h)

        output = self.linear_dropout(attn_output)

        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits

    def forward_normal_lstm(self):
        inp = self.embedding_layer(input)
        inp = self.embedding_dropout(inp)
        lstm_output, (final_h, final_c) = self.lstm(inp)
        # output.size() = (batch_size, num_seq, hidden_size)

        output = lstm_output[:, -1]
        output = self.linear_dropout(output)

        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits

    def loss(self, input, target):
        logits = self.forward(input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch
        return loss

    def predict(self, input):
        logits = self.forward(input)
        logits[:, :2] = float('-inf')
        preds = logits.max(dim=-1)[1]
        preds = preds.detach().cpu().numpy().tolist()
        return preds

    def loss_n_acc(self, input, target):
        logits = self.forward(input)
        bacth_size=input.shape[0]
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch

        pred_flat = logits_flat.max(dim=-1)[1]
        acc = (pred_flat == target_flat).sum().float()
        return loss, acc.item()
        # return loss,acc.item()/bacth_size


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=1, type=int, help='GPU nums to use')
    parser.add_argument('--epochs', default=300, type=int,metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model_path', default='./textmodel/', type=str, help='path to model')
    parser.add_argument('--snapshot', default='./pretrained/rnn_0.25325.pkl', type=str, required=False, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args
def print_args(args):
    print ("==========================================")
    print ("==========       CONFIG      =============")
    print ("==========================================")
    for arg,content in args.__dict__.items():
        print("{}:{}".format(arg,content))
    print ("\n")
def main():
    args = arg_parse()
    print_args(args)
    # if args.gpu is not None:
    #     model = model.cuda()
    #     cudnn.benchmark = True

    vector=torch.rand([500,100])
    model=LSTMClassifier(emb_vectors=vector).cuda()
    if False:# if os.path.isfile(args.snapshot):  # 'snapshot是path to latest checkpoint'
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))
        # exit()
    # opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9)
    opt = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=1.0, rho=0.9,  eps=1e-6)
    # scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    data_set = CubTextDataset('dataset', 'list/text/train.txt', 'train')
    data_loader = DataLoader(dataset=data_set,batch_size=32, shuffle=True )
    data_set1 = CubTextDataset('dataset', 'list/text/test.txt', 'test')
    data_loader1 = DataLoader(dataset=data_set1,batch_size=32, shuffle=True )
    savepath = args.model_path
    if not os.path.exists(savepath):
       os.makedirs(savepath)
    for epoch in range(args.epochs):
        sum=0
        labelsum=0
        sumx = 0
        sumy=0
        for x,y in data_loader:
            # scheduler.step()
            input=Variable(x.cuda())
            label=Variable(y.cuda())
            loss=model.loss(input,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sumx+=model.loss_n_acc(input,label)[1]
            sumy+=input.size()[0]
            image_acc=sumx/sumy
        print('epoch',epoch,model.loss(input,label),image_acc)
        save_model_path = savepath + 'epoch_' + str(epoch) + '_' + str(image_acc) + '.pkl'
        torch.save(model.state_dict(), save_model_path)
        for a, b in data_loader1:
            testa=Variable(a.cuda())
            testb=Variable(b.cuda())
            sum += model.loss_n_acc(testa, testb)[1]
            labelsum += testb.size()[0]
        print('epochtest',epoch,sum/labelsum)
if __name__ == "__main__":
    main()