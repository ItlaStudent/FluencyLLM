# ADR-0009 — Store de memoria: implementación propia en Zig

- **Contexto:** nullclaw usa SQLite (requiere libc/binding C ⇒ chocaría con Art. I). Hay que elegir store.
- **Decisión:** **store embebido escrito 100 % en Zig** (estructura simple tipo LSM/B-tree + índice invertido + almacén de vectores), **sin SQLite ni C**.
- **Consecuencias:** más trabajo de implementación, pero preserva la pureza (Art. I) y mantiene el binario diminuto. Reimplementar sólo lo mínimo necesario.
