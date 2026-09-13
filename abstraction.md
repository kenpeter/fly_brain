# abstraction

```
             CLOUD (once)                    YOUR MAC (every frame)
             ----------                      ---------------------

  +------------------+                 +------------------+
  | neuPrint API     | --fetch once--> | weight matrix W  |
  | male-cns:v1.0    |  PPL101+GNG     | 166k wiring copy |
  +------------------+                 +--------+---------+
                                                |
                                                v
+----------+   frames   +-----------+  spikes  +------------+  buttons  +------+
| NES Mario| ---------> | pixels->eye| -------> | fly brain  | --------> | game |
|  env     |            |  encode    |          |  SNN sim   |           | step |
+----------+            +-----------+          +------------+           +------+
     |                                                              |
     +---------------- reward / death ------------------------------+
                                    |
                                    v
                           +----------------+
                           | RL coach       |
                           | tweaks readout |
                           | only, W frozen |
                           +----------------+
```

One fetch builds the graph. The loop runs local. RL sits outside both.
