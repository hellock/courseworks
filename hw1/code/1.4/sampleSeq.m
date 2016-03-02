function seq = sampleSeq(pi, trans, emis, len)
%sampleSeq sample a sequence given the HMM parameters
seq.state = zeros(len, 1);
seq.observation = zeros(len, 1); 
seq.state(1) = probSample(pi);
for i = 2: len
    seq.state(i) = probSample(trans(seq.state(i-1), :));
end
for i = 1: len
    seq.observation(i) = probSample(emis(seq.state(i), :));
end
end

function r = probSample(probs)
%probSample sampling according to probabilities
accu = cumsum(probs);
rand_num = rand();
for i = 1: length(probs)
    if rand_num < accu(i)
        r = i;
        break;
    end
end
end