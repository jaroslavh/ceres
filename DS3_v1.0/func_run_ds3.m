function res = func_run_ds3(input, shape)

D = reshape(input, shape, shape);
alpha = 3;

Z = ds3(D, alpha, "R");

res = findRepresentatives(Z);
