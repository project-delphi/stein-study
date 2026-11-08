| slot | level | score (mean of seeds) | per-seed | leak_r2 (mean) | chosen |
|---|---|---|---|---|---|
| shapes/vicreg | {'lam': 1.0} | 0.268 | 0.238, 0.298 | 0.201 | |
| shapes/vicreg | {'lam': 5.0} | 0.391 | 0.410, 0.372 | 0.812 | **yes** |
| shapes/vicreg | {'lam': 25.0} | 0.319 | 0.321, 0.317 | 0.741 | |
| shapes/sigreg | {'lam': 0.01} | 0.494 | 0.512, 0.475 | 0.010 | |
| shapes/sigreg | {'lam': 0.05} | 0.537 | 0.528, 0.546 | 0.286 | **yes** |
| shapes/sigreg | {'lam': 0.25} | 0.311 | 0.328, 0.295 | 0.409 | |
| shapes/sksd | {'lam': 1.0} | 0.538 | 0.521, 0.554 | 0.029 | **yes** |
| shapes/sksd | {'lam': 5.0} | 0.460 | 0.473, 0.448 | 0.533 | |
| shapes/sksd | {'lam': 25.0} | 0.309 | 0.317, 0.301 | 0.556 | |
| shapes/ksd | {'lam': 0.2} | 0.705 | 0.721, 0.690 | 0.003 | **yes** |
| shapes/ksd | {'lam': 1.0} | 0.578 | 0.582, 0.573 | 0.578 | |
| shapes/ksd | {'lam': 5.0} | 0.324 | 0.314, 0.333 | 0.676 | |
| shapes/mmd | {'lam': 2.0} | 0.318 | 0.359, 0.278 | 0.058 | |
| shapes/mmd | {'lam': 10.0} | 0.610 | 0.623, 0.597 | 0.019 | **yes** |
| shapes/mmd | {'lam': 50.0} | 0.340 | 0.348, 0.332 | 0.379 | |
| shapes/ema | {'lam': 0.0, 'ema_tau': 0.99} | 0.393 | 0.604, 0.182 | 0.174 | |
| shapes/ema | {'lam': 0.0, 'ema_tau': 0.996} | 0.501 | 0.530, 0.471 | 0.137 | |
| shapes/ema | {'lam': 0.0, 'ema_tau': 0.999} | 0.505 | 0.630, 0.381 | 0.123 | **yes** |
| shapes/oracle | {'lam': 0.02} | 0.375 | 0.231, 0.518 | 0.159 | |
| shapes/oracle | {'lam': 0.1} | 0.438 | 0.362, 0.514 | 0.412 | **yes** |
| shapes/oracle | {'lam': 0.5} | 0.328 | 0.331, 0.325 | 0.593 | |
| shapes/ksd@uniform-prior=uniform | {'lam': 0.2} | 0.629 | 0.649, 0.609 | 0.149 | **yes** |
| shapes/ksd@uniform-prior=uniform | {'lam': 1.0} | 0.344 | 0.324, 0.363 | 0.746 | |
| shapes/ksd@uniform-prior=uniform | {'lam': 5.0} | 0.339 | 0.328, 0.351 | 0.789 | |
| shapes/ksd@laplace-prior=laplace | {'lam': 0.2} | 0.335 | 0.317, 0.352 | 0.425 | |
| shapes/ksd@laplace-prior=laplace | {'lam': 1.0} | 0.379 | 0.376, 0.381 | 0.448 | **yes** |
| shapes/ksd@laplace-prior=laplace | {'lam': 5.0} | 0.353 | 0.326, 0.381 | 0.358 | |
| shapes/sksd@uniform-prior=uniform | {'lam': 1.0} | 0.223 | 0.210, 0.236 | 0.398 | |
| shapes/sksd@uniform-prior=uniform | {'lam': 5.0} | 0.226 | 0.220, 0.232 | 0.575 | **yes** |
| shapes/sksd@uniform-prior=uniform | {'lam': 25.0} | 0.207 | 0.232, 0.183 | 0.620 | |
| shapes/sksd@laplace-prior=laplace | {'lam': 1.0} | 0.283 | 0.225, 0.342 | 0.366 | |
| shapes/sksd@laplace-prior=laplace | {'lam': 5.0} | 0.327 | 0.376, 0.278 | 0.454 | **yes** |
| shapes/sksd@laplace-prior=laplace | {'lam': 25.0} | 0.258 | 0.283, 0.234 | 0.450 | |
| shapes/sigreg@laplace | {'lam': 0.01} | 0.584 | 0.558, 0.609 | 0.004 | **yes** |
| shapes/sigreg@laplace | {'lam': 0.05} | 0.438 | 0.513, 0.363 | 0.317 | |
| shapes/sigreg@laplace | {'lam': 0.25} | 0.341 | 0.354, 0.328 | 0.490 | |
| shapes/sksd@laplace | {'lam': 1.0} | 0.629 | 0.675, 0.582 | 0.078 | **yes** |
| shapes/sksd@laplace | {'lam': 5.0} | 0.453 | 0.509, 0.397 | 0.562 | |
| shapes/sksd@laplace | {'lam': 25.0} | 0.328 | 0.308, 0.347 | 0.547 | |
| shapes/ksd@laplace | {'lam': 0.2} | 0.853 | 0.850, 0.856 | 0.243 | **yes** |
| shapes/ksd@laplace | {'lam': 1.0} | 0.549 | 0.547, 0.551 | 0.632 | |
| shapes/ksd@laplace | {'lam': 5.0} | 0.433 | 0.422, 0.443 | 0.848 | |
| shapes/mmd@laplace | {'lam': 2.0} | 0.471 | 0.526, 0.416 | 0.044 | |
| shapes/mmd@laplace | {'lam': 10.0} | 0.577 | 0.701, 0.453 | 0.098 | **yes** |
| shapes/mmd@laplace | {'lam': 50.0} | 0.362 | 0.414, 0.310 | 0.368 | |
| shapes/oracle@laplace | {'lam': 0.02} | 0.524 | 0.443, 0.605 | 0.141 | **yes** |
| shapes/oracle@laplace | {'lam': 0.1} | 0.518 | 0.518, 0.519 | 0.518 | |
| shapes/oracle@laplace | {'lam': 0.5} | 0.382 | 0.353, 0.411 | 0.570 | |
| temporal/ema | {'lam': 0.0, 'ema_tau': 0.99} | 0.320 | 0.286, 0.353 | 0.377 | **yes** |
| temporal/ema | {'lam': 0.0, 'ema_tau': 0.996} | 0.234 | 0.242, 0.226 | 0.456 | |
| temporal/ema | {'lam': 0.0, 'ema_tau': 0.999} | 0.300 | 0.284, 0.316 | 0.767 | |
| temporal/ema+sksd | {'lam': 1.0} | 0.254 | 0.258, 0.249 | 0.689 | |
| temporal/ema+sksd | {'lam': 5.0} | 0.290 | 0.296, 0.284 | 0.690 | **yes** |
| temporal/ema+sksd | {'lam': 25.0} | 0.277 | 0.282, 0.272 | 0.601 | |
| temporal/ema+sigreg | {'lam': 0.01} | 0.194 | 0.196, 0.192 | 0.526 | |
| temporal/ema+sigreg | {'lam': 0.05} | 0.214 | 0.230, 0.198 | 0.548 | **yes** |
| temporal/ema+sigreg | {'lam': 0.25} | 0.201 | 0.200, 0.203 | 0.413 | |
| temporal/sksd | {'lam': 1.0} | 0.553 | 0.581, 0.525 | 0.106 | **yes** |
| temporal/sksd | {'lam': 5.0} | 0.502 | 0.501, 0.504 | 0.518 | |
| temporal/sksd | {'lam': 25.0} | 0.318 | 0.331, 0.305 | 0.632 | |
| temporal/vicreg | {'lam': 1.0} | 0.408 | 0.420, 0.395 | 0.453 | **yes** |
| temporal/vicreg | {'lam': 5.0} | 0.390 | 0.391, 0.389 | 0.739 | |
| temporal/vicreg | {'lam': 25.0} | 0.328 | 0.332, 0.325 | 0.760 | |
| cifar/vicreg | {'lam': 1.0} | 0.272 | 0.270, 0.275 | 0.001 | |
| cifar/vicreg | {'lam': 5.0} | 0.330 | 0.324, 0.336 | 0.020 | **yes** |
| cifar/vicreg | {'lam': 25.0} | 0.321 | 0.320, 0.322 | 0.056 | |
| cifar/sigreg | {'lam': 0.01} | 0.320 | 0.319, 0.320 | 0.005 | |
| cifar/sigreg | {'lam': 0.05} | 0.338 | 0.334, 0.341 | 0.027 | **yes** |
| cifar/sigreg | {'lam': 0.25} | 0.268 | 0.265, 0.271 | 0.022 | |
| cifar/sksd | {'lam': 1.0} | 0.305 | 0.314, 0.297 | 0.006 | |
| cifar/sksd | {'lam': 5.0} | 0.343 | 0.344, 0.342 | 0.033 | **yes** |
| cifar/sksd | {'lam': 25.0} | 0.284 | 0.289, 0.279 | 0.041 | |
| cifar/ksd | {'lam': 0.2} | 0.364 | 0.362, 0.367 | 0.023 | **yes** |
| cifar/ksd | {'lam': 1.0} | 0.334 | 0.332, 0.336 | 0.054 | |
| cifar/ksd | {'lam': 5.0} | 0.285 | 0.278, 0.291 | 0.050 | |
| cifar/mmd | {'lam': 2.0} | 0.272 | 0.266, 0.279 | 0.000 | |
| cifar/mmd | {'lam': 10.0} | 0.343 | 0.347, 0.339 | 0.021 | **yes** |
| cifar/mmd | {'lam': 50.0} | 0.307 | 0.310, 0.304 | 0.033 | |
| cifar/ema | {'lam': 0.0, 'ema_tau': 0.99} | 0.316 | 0.336, 0.296 | 0.009 | |
| cifar/ema | {'lam': 0.0, 'ema_tau': 0.996} | 0.324 | 0.360, 0.287 | 0.032 | |
| cifar/ema | {'lam': 0.0, 'ema_tau': 0.999} | 0.346 | 0.350, 0.343 | 0.063 | **yes** |
| vae/betavae | {'lam': 0.2} | 0.328 | 0.251, 0.406 | 0.789 | **yes** |
| vae/betavae | {'lam': 1.0} | 0.211 | 0.226, 0.196 | 0.500 | |
| vae/betavae | {'lam': 5.0} | 0.204 | 0.193, 0.215 | 0.361 | |
| vae/ksd | {'lam': 2.0} | 0.316 | 0.322, 0.311 | 0.967 | |
| vae/ksd | {'lam': 10.0} | 0.333 | 0.358, 0.308 | 0.961 | **yes** |
| vae/ksd | {'lam': 50.0} | 0.327 | 0.346, 0.308 | 0.889 | |
| vae/mmd | {'lam': 20.0} | 0.336 | 0.348, 0.324 | 0.975 | **yes** |
| vae/mmd | {'lam': 100.0} | 0.324 | 0.291, 0.356 | 0.939 | |
| vae/mmd | {'lam': 500.0} | 0.316 | 0.318, 0.315 | 0.790 | |
| vae/ksd@laplace-prior=laplace | {'lam': 2.0} | 0.495 | 0.492, 0.498 | 0.977 | **yes** |
| vae/ksd@laplace-prior=laplace | {'lam': 10.0} | 0.379 | 0.430, 0.328 | 0.864 | |
| vae/ksd@laplace-prior=laplace | {'lam': 50.0} | 0.325 | 0.380, 0.270 | 0.865 | |
| vae/ksd@laplace | {'lam': 2.0} | 0.519 | 0.483, 0.554 | 0.981 | **yes** |
| vae/ksd@laplace | {'lam': 10.0} | 0.445 | 0.420, 0.471 | 0.931 | |
| vae/ksd@laplace | {'lam': 50.0} | 0.427 | 0.400, 0.453 | 0.978 | |
