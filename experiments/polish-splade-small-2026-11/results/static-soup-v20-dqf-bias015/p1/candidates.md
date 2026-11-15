| table | tiny10 | vs released | heldout8 | dims/query | postings/query | arguana-pl: dims / postings | other tasks: postings | within 16 807 | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `phr2-k8-kd-r2` | 60.74 | 0.0 | 39.24 | 89.7 | 11764 | 375.9 / 43673 | 7775 | yes | the released table (128 query tokens) |
| `phr2-k8-kd-r2-len512` | 60.74 | 0.0 |  | 118.2 | 14253 | 632.2 / 66078 | 7775 | yes | reads 512 query tokens, rows unchanged |
| `a1-sparseadam` | 60.79 | 0.05 |  | 94.0 | 12258 | 394.8 / 45727 | 8074 | yes | A1 arm sparseadam from phr2-k8-kd (--optimizer sparse_adam) |
| `a1-adagrad` | 60.75 | 0.01 |  | 98.0 | 12780 | 410.7 / 47632 | 8424 | yes | A1 arm adagrad from phr2-k8-kd (--optimizer adagrad --lr 5e-2) |
| `a1-floor` | 60.77 | 0.03 |  | 133.8 | 17682 | 521.1 / 60613 | 12316 | NO | A1 arm floor from phr2-k8-kd (--value-floor 1e-4) |
| `a1-sparseadam-mw005` | 60.8 | 0.06 |  | 92.4 | 12091 | 387.5 / 44991 | 7979 | yes | a1-sparseadam with table weights below 0.05 dropped |
| `a1-sparseadam-mw010` | 60.8 | 0.06 |  | 88.4 | 11271 | 372.5 / 43202 | 7280 | yes | a1-sparseadam with table weights below 0.10 dropped |
| `a1-sparseadam-mw020` | 60.74 | 0.0 |  | 77.5 | 9881 | 328.2 / 38325 | 6326 | yes | a1-sparseadam with table weights below 0.20 dropped |
| `a1-sparseadam-mw005-g1-kd` | 60.71 | -0.03 |  | 97.6 | 12050 | 410.5 / 46282 | 7771 | yes | A2 round 1: up to 2 new slots per row of a1-sparseadam-mw005, 3 epochs on pool r2 |

Best within the limit: `a1-sparseadam-mw005`. Released table: `phr2-k8-kd-r2` (60.74 / 39.24). Full model symmetric: 65.06 / 43.64, 123.0 dims/query, 13 945 postings/query.
