# Execution results based on the optimization steps in [faster_matmul/readme](README.md)

All results are executed on one **Nvidia A6000** GPU instance, *natively*.

1. Naive
    
    `SIZE: 2048; average elapsed time: 0.0565261s; performance: 303.928 GFLOPS`
    
    `SIZE: 4096; average elapsed time: 0.449229s; performance: 305.944 GFLOPS`

2. Coalesced Memory Access

    `SIZE: 2048; average elapsed time: 0.00982509s; performance: 1748.57 GFLOPS`

    `SIZE: 4096; average elapsed time: 0.0872624s; performance: 1575.01 GFLOPS`

3. Shared Memory Blocking

    `SIZE: 2048; average elapsed time: 0.00578487s; performance: 2969.79 GFLOPS`

    `SIZE: 4096; average elapsed time: 0.0456818s; performance: 3008.61 GFLOPS`

4. 1D Blocktiling

    `SIZE: 2048; average elapsed time: 0.00208048s; performance: 8257.66 GFLOPS`

    `SIZE: 4096; average elapsed time: 0.0162718s; performance: 8446.47 GFLOPS`

