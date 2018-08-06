def test_LSTM(sentences, vocab, eos_separator, settings, cuda):
    import torch
    import lstm
    from tqdm import tqdm
    import numpy as np
    import os.path as op

    model = torch.load(op.join(settings.path2LSTMdata, settings.LSTM_pretrained_model))
    model.rnn.flatten_parameters()
    # hack the forward function to send an extra argument containing the model parameters
    model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)

    # output buffers
    fixed_length_arrays = False
    if fixed_length_arrays:
        vectors = np.zeros((len(sentences), 2 *model.nhid*model.nlayers, max_length))
        log_probabilities =  np.zeros((len(sentences), max_length))
        gates = {k: np.zeros((len(sentences), model.nhid*model.nlayers, max_length)) for k in ['in', 'forget', 'out', 'c_tilde']}
    else:
        # vectors = [np.zeros((2*model.nhid*model.nlayers, len(s))) for s in tqdm(sentences)] #np.zeros((len(sentences), 2 *model.nhid*model.nlayers, max_length))
        # log_probabilities = [np.zeros(len(s)) for s in tqdm(sentences)] # np.zeros((len(sentences), max_length))
        # gates = {k: [np.zeros((model.nhid*model.nlayers, len(s))) for s in tqdm(sentences)] for k in ['in', 'forget', 'out', 'c_tilde']} #np.zeros((len(sentences), model.nhid*model.nlayers, max_length)) for k in ['in', 'forget', 'out', 'c_tilde']}
        # vectors = [np.zeros((2 * model.nhid * model.nlayers, len(s))) for s in sentences]
        vectors = [np.zeros((model.nhid * model.nlayers, len(s))) for s in sentences]
        # np.zeros((len(sentences), 2 *model.nhid*model.nlayers, max_length))
        # log_probabilities = [np.zeros(len(s)) for s in sentences]  # np.zeros((len(sentences), max_length))
        # gates = {k: [np.zeros((model.nhid * model.nlayers, len(s))) for s in sentences] for k in
        #          ['in', 'forget', 'out',
        #           'c_tilde']}  # np.zeros((len(sentences), model.nhid*model.nlayers, max_length)) for k in ['in', 'forget', 'out', 'c_tilde']}

    # for i, s in enumerate(tqdm(sentences)):
    for i, s in enumerate(sentences):
        if i%100==1: print(str(i) +' out of ' + str(len(sentences)))
        #sys.stdout.write("{}% complete ({} / {})\r".format(int(i/len(sentences) * 100), i, len(sentences)))
        out = None
        # reinit hidden
        hidden = model.init_hidden(1)
        # intitialize with end of sentence
        inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[eos_separator]]]))
        if cuda:
            inp = inp.cuda()
        _, hidden = model(inp, hidden)
        # out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
        for j, w in enumerate(s):
            # store the surprisal for the current word
            # log_probabilities[i][j] = out[0,0,vocab.word2idx[w]].data[0]

            inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
            if cuda:
                inp = inp.cuda()
            _, hidden = model(inp, hidden)
            # out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)

            # vectors[i][:,j] = torch.cat([h.data.view(1,1,-1) for h in hidden],2).cpu().numpy()
            # vectors[i][:, j] = hidden[0].data.view(1, 1, -1).cpu().numpy() # Take only 1st hidden (h?)
            vectors[i][:, j] = hidden[settings.h_or_c].data.view(1, 1, -1).cpu().numpy()  # Take only 1st or 2nd hidden (h or c?)

        np.append(vectors[i], np.expand_dims(np.arange(len(s)) + 1, axis=0), axis=0)
            # we can retrieve the gates thanks to the hacked function
            # for k, gates_k in gates.items():
            #     gates_k[i][:,j] = torch.cat([g[k].data for g in model.rnn.last_gates],1).cpu().numpy()


        # out = {
        #     'vectors': vectors,
        #     'log_probabilities': log_probabilities
        # }

        # for k, g in gates.items():
        #     out['gates.{}'.format(k)] = g

    return vectors