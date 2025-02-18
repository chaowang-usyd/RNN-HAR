function indx = utils_rs_multinomial(w)  % W = 1*2000

N = length(w); % number of particles
indx = zeros(1,N); % preallocate 
Q = cumsum(w); % cumulative sum, 1*2000
u = sort(rand(1,N)); % random numbers, 1*2000

j = 1;
for i=1:N  % 1 : 2000
    while (Q(j)<u(i))
        j = j+1; % climb the ladder
    end
    indx(i) = j; % assign index
end
