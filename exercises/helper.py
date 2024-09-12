import pandas as pd


def get_title(df: pd.DataFrame, f_m: float, h_M: float, h_m: float) -> str:
    g = [r"f$_{m}$=<fm>".replace("<fm>", f"{f_m}")]
    if h_M == h_m:
        s = r"h$_{M}$=h$_{m}$=<h>".replace("<h>", f"{h_M}")
    else:
        s = r"h$_{M}$>h$_{m}$" if h_M > h_m else r"h$_{M}$<h$_{m}$"
    g.append(s)
    return f"{df.name}\n{', '.join(g)}"
