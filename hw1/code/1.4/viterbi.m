function s = viterbi( pi, trans, emis, x )
%viterbi Viterbi Algorithm for decoding HMM
s = zeros(size(x));
c = length(pi);
T = length(x);
phi = zeros(T, c);
bp = zeros(T, c);
phi(1, :) = pi .* emis(:, x(1))';
for t = 2: T
    for j = 1: c
        [phi(t, j), bp(t, j)] = max(phi(t-1, :) .* trans(:, j)' * emis(j, x(t)));
    end
end
[~, idx] = max(phi(T, :));
s(T) = bp(T, idx);
for t = T: -1: 2
    s(t-1) = bp(t, s(t));
end
end

