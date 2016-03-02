%% set params
pi = [0.6, 0.4];
trans = [0.7, 0.3;
         0.4, 0.6];
emission = [0.1, 0.4, 0.5;
            0.6, 0.3, 0.1];
states = [1, 2];
observations = [1, 2, 3];
%% generate sequence
seq = sampleSeq(pi, trans, emission, 10);
seq.state = states(seq.state);
seq.observation = observations(seq.observation);
%% decode
decode_seq = viterbi(pi, trans, emission, seq.observation);