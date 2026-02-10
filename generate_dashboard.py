
from tvscreener import StockScreener, StockField, IndexSymbol
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')


ss = StockScreener()
ss.set_index(IndexSymbol.SP500) 
ss.select(
    StockField.NAME,
    StockField.PRICE,
    StockField.CHANGE_PERCENT,
    StockField.VOLUME,
    StockField.AVERAGE_VOLUME_30D_CALC_1,
    StockField.SECTOR,
    StockField.INDUSTRY,
    StockField.MARKET_CAPITALIZATION,
)
ss.set_range(0, 500)
df = ss.get()
df = df.dropna(subset=['Change %'])

# Detectar el nombre real de la columna de volumen promedio
vol_col = None
for c in df.columns:
    if 'average' in c.lower() and 'volume' in c.lower():
        vol_col = c
        break


total_stocks  = len(df)
advances      = len(df[df['Change %'] > 0])
declines      = len(df[df['Change %'] < 0])
unchanged     = total_stocks - advances - declines
ad_ratio      = advances / declines if declines > 0 else float('inf')
avg_change    = df['Change %'].mean()
median_change = df['Change %'].median()
std_change    = df['Change %'].std()

top_gainers = df.nlargest(20, 'Change %')[['Name', 'Sector', 'Industry', 'Change %', 'Price', 'Volume']].copy()
top_losers  = df.nsmallest(20, 'Change %')[['Name', 'Sector', 'Industry', 'Change %', 'Price', 'Volume']].copy()

# Outliers por volumen
has_volume_data  = False
volume_outliers  = pd.DataFrame()

if vol_col:
    df_vol = df.dropna(subset=[vol_col, 'Volume']).copy()
    df_vol[vol_col] = pd.to_numeric(df_vol[vol_col], errors='coerce')
    df_vol = df_vol.dropna(subset=[vol_col])
    df_vol = df_vol[df_vol[vol_col] > 0]
    df_vol['Volume Ratio'] = df_vol['Volume'] / df_vol[vol_col]
    volume_outliers = df_vol[df_vol['Volume Ratio'] >= 2.0].nlargest(20, 'Volume Ratio')[
        ['Name', 'Sector', 'Change %', 'Volume', vol_col, 'Volume Ratio']
    ].copy()
    volume_outliers['Volume Ratio'] = volume_outliers['Volume Ratio'].round(2)
    volume_outliers[vol_col] = volume_outliers[vol_col].astype(int)
    volume_outliers = volume_outliers.rename(columns={vol_col: 'Vol. Prom. 30d'})
    if not volume_outliers.empty:
        has_volume_data = True

# Promedios por sector
sector_stats = df.groupby('Sector').agg(
    avg_change=('Change %', 'mean'),
    count=('Change %', 'count'),
    advances=('Change %', lambda x: (x > 0).sum()),
    declines=('Change %', lambda x: (x < 0).sum()),
).reset_index()
sector_stats = sector_stats[sector_stats['count'] >= 3].sort_values('avg_change', ascending=False)

# Sentimiento
if ad_ratio >= 2.0 and avg_change >= 0.5:
    market_sentiment = "Alcista amplio"
    sentiment_color  = "#22c55e"
elif ad_ratio >= 1.2 and avg_change >= 0:
    market_sentiment = "Alcista moderado"
    sentiment_color  = "#86efac"
elif ad_ratio <= 0.5 and avg_change <= -0.5:
    market_sentiment = "Bajista amplio"
    sentiment_color  = "#ef4444"
elif ad_ratio <= 0.8 and avg_change <= 0:
    market_sentiment = "Bajista moderado"
    sentiment_color  = "#fca5a5"
else:
    market_sentiment = "Mixto / Sin tendencia clara"
    sentiment_color  = "#94a3b8"

leading_sector = sector_stats.iloc[0]['Sector']  if not sector_stats.empty else "N/D"
lagging_sector = sector_stats.iloc[-1]['Sector'] if not sector_stats.empty else "N/D"

def build_summary():
    lines = []
    lines.append(
        f"En la jornada analizada, el universo de {total_stocks} acciones registró un avance/caída "
        f"de {advances}/{declines} (ratio {ad_ratio:.2f}), con un cambio promedio de {avg_change:+.2f}%."
    )
    if ad_ratio >= 1.5:
        lines.append(
            "La amplitud del mercado es positiva: más de la mitad de los instrumentos terminaron "
            "en verde, lo que indica un movimiento de base amplia y no concentrado en pocos nombres."
        )
    elif ad_ratio <= 0.67:
        lines.append(
            "La amplitud del mercado es negativa: la mayoría de los instrumentos cedieron terreno, "
            "lo cual sugiere un deterioro generalizado y no puntual."
        )
    else:
        lines.append(
            "La amplitud del mercado es mixta: el movimiento del día no muestra una dirección "
            "dominante clara, lo que suele indicar rotación sectorial o falta de catalizadores definidos."
        )
    lines.append(
        f"El sector con mejor desempeño promedio fue {leading_sector}, "
        f"mientras que el de peor desempeño fue {lagging_sector}."
    )
    lines.append(
        f"La dispersión de los cambios (desvío estándar: {std_change:.2f}%) "
        + ("es elevada, lo que refleja alta selectividad entre instrumentos."
           if std_change > 2 else
           "es moderada, con movimientos relativamente homogéneos en el universo.")
    )
    if has_volume_data:
        lines.append(
            f"Se detectaron {len(volume_outliers)} acciones con volumen significativamente superior "
            f"al promedio de 30 ruedas (ratio ≥ 2x), lo que puede anticipar continuidad o reversión "
            f"de tendencia en esos instrumentos."
        )
    return " ".join(lines)

executive_summary = build_summary()


PALETTE_POS  = "#3b82f6"
PALETTE_NEG  = "#f97316"
PALETTE_NEUT = "#94a3b8"
BG           = "#f8fafc"

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_b64

# Gráfico 1: Top Ganadores / Perdedores
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
for ax, data, title in [
    (axes1[0], top_gainers, 'Top 20 Ganadores'),
    (axes1[1], top_losers,  'Top 20 Perdedores'),
]:
    colores = [PALETTE_POS if v > 0 else PALETTE_NEG for v in data['Change %']]
    ax.barh(range(len(data)), data['Change %'], color=colores, alpha=0.85, edgecolor='none')
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(
        [f"{row['Name']}  ·  {row['Sector']}" for _, row in data.iterrows()],
        fontsize=8, fontfamily='monospace'
    )
    ax.set_xlabel('Cambio %', fontsize=10, labelpad=8)
    ax.set_title(title, fontweight='bold', fontsize=13, pad=12)
    ax.invert_yaxis()
    ax.axvline(0, color='#334155', linewidth=0.8)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
plt.tight_layout(pad=2.5)
img_movers = fig_to_base64(fig1)
plt.close(fig1)

# Gráfico 2: Heatmap de sectores
fig2, ax2 = plt.subplots(figsize=(14, 6), facecolor=BG)
n       = len(sector_stats)
cols_hm = min(n, 6)
rows_hm = int(np.ceil(n / cols_hm))
vmax    = max(abs(sector_stats['avg_change'].max()), abs(sector_stats['avg_change'].min()))
cmap    = plt.cm.RdYlGn
cell_w, cell_h = 1.0, 0.7

for i, row_s in sector_stats.reset_index(drop=True).iterrows():
    col_i  = i % cols_hm
    row_i  = i // cols_hm
    val    = row_s['avg_change']
    norm_v = (val + vmax) / (2 * vmax) if vmax != 0 else 0.5
    color  = cmap(norm_v)
    rect   = mpatches.FancyBboxPatch(
        (col_i * (cell_w + 0.08), -(row_i * (cell_h + 0.08))),
        cell_w, cell_h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor='white', linewidth=1.5
    )
    ax2.add_patch(rect)
    cx = col_i * (cell_w + 0.08) + cell_w / 2
    cy = -(row_i * (cell_h + 0.08)) + cell_h / 2
    text_color = 'white' if abs(norm_v - 0.5) > 0.2 else '#1e293b'
    ax2.text(cx, cy + 0.12, row_s['Sector'], ha='center', va='center',
             fontsize=8, fontweight='bold', color=text_color)
    ax2.text(cx, cy - 0.12, f"{val:+.2f}%  ({int(row_s['count'])})", ha='center', va='center',
             fontsize=7.5, color=text_color, alpha=0.9)

ax2.set_xlim(-0.1, cols_hm * (cell_w + 0.08))
ax2.set_ylim(-(rows_hm * (cell_h + 0.08)) + 0.05, cell_h + 0.1)
ax2.axis('off')
ax2.set_title('Heatmap de Sectores  —  Cambio Promedio del Día', fontweight='bold', fontsize=13, pad=14)
ax2.set_facecolor(BG)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax2, orientation='horizontal', fraction=0.03, pad=0.02, aspect=40)
cbar.set_label('Cambio promedio %', fontsize=9)
plt.tight_layout()
img_heatmap = fig_to_base64(fig2)
plt.close(fig2)

# Gráfico 3: Histograma + Market Breadth
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
bins = np.linspace(df['Change %'].quantile(0.01), df['Change %'].quantile(0.99), 45)
ax3a.hist(df['Change %'], bins=bins, color=PALETTE_POS, alpha=0.75, edgecolor='white', linewidth=0.4)
ax3a.axvline(avg_change,    color='#1e293b',   linewidth=1.8, linestyle='--', label=f'Media: {avg_change:+.2f}%')
ax3a.axvline(median_change, color=PALETTE_NEG, linewidth=1.5, linestyle=':',  label=f'Mediana: {median_change:+.2f}%')
ax3a.axvline(avg_change + std_change, color='#64748b', linewidth=1, linestyle='--', alpha=0.5)
ax3a.axvline(avg_change - std_change, color='#64748b', linewidth=1, linestyle='--', alpha=0.5, label=f'±1 SD: {std_change:.2f}%')
ax3a.set_xlabel('Cambio %', fontsize=10)
ax3a.set_ylabel('Cantidad de acciones', fontsize=10)
ax3a.set_title('Distribución de Cambios del Día', fontweight='bold', fontsize=13)
ax3a.legend(fontsize=8)
ax3a.grid(axis='y', alpha=0.2, linestyle='--')
ax3a.set_facecolor(BG)
for spine in ax3a.spines.values():
    spine.set_visible(False)

ad_data   = [advances, unchanged, declines]
ad_colors = [PALETTE_POS, PALETTE_NEUT, PALETTE_NEG]
ad_labels = [f'Alcistas\n{advances}', f'Sin cambio\n{unchanged}', f'Bajistas\n{declines}']
left = 0
for val, col, lbl in zip(ad_data, ad_colors, ad_labels):
    ax3b.barh(0, val, left=left, color=col, alpha=0.85, height=0.5)
    if val > 5:
        ax3b.text(left + val / 2, 0, lbl, ha='center', va='center',
                  fontsize=9, fontweight='bold', color='white')
    left += val
ax3b.set_xlim(0, total_stocks)
ax3b.set_ylim(-1, 1)
ax3b.axis('off')
ax3b.set_title(f'Market Breadth  —  Ratio A/D: {ad_ratio:.2f}', fontweight='bold', fontsize=13)
ax3b.text(total_stocks / 2, -0.55,
          f'{advances} suben  ·  {unchanged} sin cambio  ·  {declines} bajan  (universo: {total_stocks})',
          ha='center', fontsize=8.5, color='#475569')
ax3b.set_facecolor(BG)
plt.tight_layout(pad=2)
img_breadth = fig_to_base64(fig3)
plt.close(fig3)

# Gráfico 4: Outliers de volumen
img_volume = None
if has_volume_data:
    fig4, ax4 = plt.subplots(figsize=(14, 6), facecolor=BG)
    colors_vol = [PALETTE_POS if v > 0 else PALETTE_NEG for v in volume_outliers['Change %']]
    ax4.barh(range(len(volume_outliers)), volume_outliers['Volume Ratio'],
             color=colors_vol, alpha=0.82, edgecolor='none')
    ax4.set_yticks(range(len(volume_outliers)))
    ax4.set_yticklabels(
        [f"{row['Name']}  ·  {row['Sector']}  ({row['Change %']:+.2f}%)"
         for _, row in volume_outliers.iterrows()],
        fontsize=8, fontfamily='monospace'
    )
    ax4.axvline(2, color='#475569', linewidth=1, linestyle='--', alpha=0.6)
    ax4.set_xlabel('Ratio Volumen / Promedio 30 ruedas', fontsize=10)
    ax4.set_title('Acciones con Volumen Inusual (≥ 2x promedio 30 ruedas)', fontweight='bold', fontsize=13)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.2, linestyle='--')
    ax4.set_facecolor(BG)
    for spine in ax4.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    img_volume = fig_to_base64(fig4)
    plt.close(fig4)



def df_to_html_table(data, table_id):
    return data.to_html(classes='data-table', table_id=table_id, escape=False, index=False)

def format_change(val):
    try:
        v = float(val)
        cls = 'positive' if v >= 0 else 'negative'
        return f'<span class="{cls}">{v:+.2f}%</span>'
    except:
        return val

for col_df in [top_gainers, top_losers]:
    col_df['Change %'] = col_df['Change %'].apply(format_change)

if has_volume_data:
    volume_outliers['Change %'] = volume_outliers['Change %'].apply(format_change)

volume_section = ""
if has_volume_data and img_volume:
    volume_table  = df_to_html_table(volume_outliers, 'volume-outliers')
    volume_section = f"""
    <section class="card full-width">
      <div class="card-header">
        <span class="card-title">Acciones con Volumen Inusual</span>
        <span class="badge">≥ 2x promedio 30 ruedas</span>
      </div>
      <div class="chart-wrap">
        <img src="data:image/png;base64,{img_volume}" alt="Volumen inusual">
      </div>
      <div class="table-scroll">{volume_table}</div>
      <div class="explainer">
        <strong>Qué muestra:</strong> Acciones cuyo volumen del día supera al menos el doble
        de su promedio de las últimas 30 ruedas. Lecturas elevadas de este ratio (3x, 5x o más)
        suelen preceder movimientos de precio importantes o confirmar rupturas técnicas.
        El color de la barra indica si el instrumento sube (azul) o baja (naranja) en la jornada,
        lo que permite leer si el volumen acompaña una presión compradora o vendedora.
      </div>
    </section>
    """



now_str  = datetime.utcnow().strftime('%Y-%m-%d  %H:%M UTC')
date_str = datetime.utcnow().strftime('%Y-%m-%d')

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="1800">
  <title>Dashboard de Análisis de Acciones</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:        #f1f5f9;
      --surface:   #ffffff;
      --border:    #e2e8f0;
      --text-main: #0f172a;
      --text-muted:#64748b;
      --blue:      #3b82f6;
      --orange:    #f97316;
      --green:     #22c55e;
      --red:       #ef4444;
      --slate:     #94a3b8;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'DM Sans', sans-serif;
      background: var(--bg);
      color: var(--text-main);
      padding: 28px 20px 60px;
      line-height: 1.6;
    }}
    .page-wrap {{ max-width: 1380px; margin: 0 auto; }}
    .page-header {{
      display: flex; justify-content: space-between; align-items: flex-end;
      margin-bottom: 32px; padding-bottom: 20px; border-bottom: 2px solid var(--border);
    }}
    .page-header h1 {{ font-size: 1.9rem; font-weight: 600; letter-spacing: -0.03em; }}
    .page-header h1 span {{ color: var(--blue); }}
    .meta {{ font-size: 0.82rem; color: var(--text-muted); font-family: 'DM Mono', monospace; text-align: right; }}
    .sentiment-banner {{
      display: flex; align-items: center; gap: 14px;
      background: var(--surface); border: 1px solid var(--border);
      border-left: 5px solid {sentiment_color};
      border-radius: 10px; padding: 16px 22px; margin-bottom: 28px;
    }}
    .sentiment-label {{ font-weight: 600; font-size: 1rem; color: {sentiment_color}; white-space: nowrap; }}
    .sentiment-text {{ color: var(--text-muted); font-size: 0.87rem; }}
    .kpi-strip {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px; margin-bottom: 28px;
    }}
    .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }}
    .kpi-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 6px; }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 600; font-family: 'DM Mono', monospace; }}
    .kpi-value.pos {{ color: var(--blue); }}
    .kpi-value.neg {{ color: var(--orange); }}
    .kpi-value.neu {{ color: var(--text-main); }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-bottom: 22px; }}
    .full-width {{ margin-bottom: 22px; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
    .card-header {{
      display: flex; align-items: center; gap: 10px;
      padding: 18px 22px 12px; border-bottom: 1px solid var(--border);
    }}
    .card-title {{ font-size: 1rem; font-weight: 600; }}
    .badge {{
      margin-left: auto; font-size: 0.72rem; font-family: 'DM Mono', monospace;
      background: #eff6ff; color: var(--blue); border: 1px solid #bfdbfe;
      border-radius: 99px; padding: 2px 10px;
    }}
    .chart-wrap {{ padding: 18px 18px 10px; }}
    .chart-wrap img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
    .explainer {{
      font-size: 0.83rem; color: var(--text-muted);
      padding: 12px 22px 18px; border-top: 1px dashed var(--border); line-height: 1.65;
    }}
    .explainer strong {{ color: var(--text-main); }}
    .summary-card {{
      background: #f8fafc; border: 1px solid var(--border); border-left: 4px solid var(--blue);
      border-radius: 10px; padding: 20px 24px; margin-bottom: 28px;
      font-size: 0.9rem; line-height: 1.75; color: #334155;
    }}
    .summary-title {{
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
      color: var(--blue); font-weight: 600; margin-bottom: 10px;
    }}
    .table-scroll {{ overflow-x: auto; padding: 0 18px 18px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; font-family: 'DM Mono', monospace; }}
    .data-table th {{
      background: #f1f5f9; color: var(--text-muted); font-size: 0.72rem;
      text-transform: uppercase; letter-spacing: 0.05em; padding: 10px 12px;
      text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap;
    }}
    .data-table td {{ padding: 9px 12px; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tr:hover td {{ background: #f8fafc; }}
    .positive {{ color: var(--blue); font-weight: 600; }}
    .negative {{ color: var(--orange); font-weight: 600; }}
    .page-footer {{
      text-align: center; margin-top: 48px; font-size: 0.78rem;
      color: var(--slate); font-family: 'DM Mono', monospace;
    }}
    @media (max-width: 860px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
      .page-header {{ flex-direction: column; align-items: flex-start; gap: 8px; }}
    }}
  </style>
</head>
<body>
<div class="page-wrap">

  <header class="page-header">
    <h1>Dashboard de <span>Análisis de Acciones</span></h1>
    <div class="meta">Universo: {total_stocks} instrumentos<br>Actualizado: {now_str}</div>
  </header>

  <div class="sentiment-banner">
    <span class="sentiment-label">{market_sentiment}</span>
    <span class="sentiment-text">
      Sector líder: <strong>{leading_sector}</strong> &nbsp;·&nbsp;
      Sector rezagado: <strong>{lagging_sector}</strong> &nbsp;·&nbsp;
      Ratio A/D: <strong>{ad_ratio:.2f}</strong> &nbsp;·&nbsp;
      Media: <strong>{avg_change:+.2f}%</strong>
    </span>
  </div>

  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-label">Universo</div><div class="kpi-value neu">{total_stocks}</div></div>
    <div class="kpi"><div class="kpi-label">Alcistas</div><div class="kpi-value pos">{advances}</div></div>
    <div class="kpi"><div class="kpi-label">Bajistas</div><div class="kpi-value neg">{declines}</div></div>
    <div class="kpi"><div class="kpi-label">Ratio A/D</div><div class="kpi-value {'pos' if ad_ratio >= 1 else 'neg'}">{ad_ratio:.2f}</div></div>
    <div class="kpi"><div class="kpi-label">Cambio promedio</div><div class="kpi-value {'pos' if avg_change >= 0 else 'neg'}">{avg_change:+.2f}%</div></div>
    <div class="kpi"><div class="kpi-label">Mediana</div><div class="kpi-value {'pos' if median_change >= 0 else 'neg'}">{median_change:+.2f}%</div></div>
    <div class="kpi"><div class="kpi-label">Desvío estándar</div><div class="kpi-value neu">{std_change:.2f}%</div></div>
  </div>

  <div class="summary-card">
    <div class="summary-title">Resumen ejecutivo</div>
    {executive_summary}
  </div>

  <div class="grid-2">
    <section class="card">
      <div class="card-header"><span class="card-title">Top 20 Ganadores</span><span class="badge">por % cambio</span></div>
      <div class="table-scroll">{df_to_html_table(top_gainers, 'gainers')}</div>
      <div class="explainer">
        <strong>Qué muestra:</strong> Las 20 acciones con mayor subida porcentual en la jornada.
        La columna Sector permite identificar si la suba es puntual o responde a una dinámica
        sectorial más amplia. Útil para detectar catalizadores específicos (resultados, noticias)
        o rotaciones en curso.
      </div>
    </section>
    <section class="card">
      <div class="card-header"><span class="card-title">Top 20 Perdedores</span><span class="badge">por % cambio</span></div>
      <div class="table-scroll">{df_to_html_table(top_losers, 'losers')}</div>
      <div class="explainer">
        <strong>Qué muestra:</strong> Las 20 acciones con mayor caída porcentual en la jornada.
        Analizar si los perdedores pertenecen al mismo sector ayuda a distinguir entre un evento
        idiosincrático (un solo nombre) y una presión vendedora sistémica sobre una industria
        o sector determinado.
      </div>
    </section>
  </div>

  <section class="card full-width">
    <div class="card-header"><span class="card-title">Heatmap de Sectores</span><span class="badge">cambio promedio del día</span></div>
    <div class="chart-wrap"><img src="data:image/png;base64,{img_heatmap}" alt="Heatmap sectores"></div>
    <div class="explainer">
      <strong>Qué muestra:</strong> Cada celda representa un sector del mercado coloreado según
      su cambio porcentual promedio en el día: verde indica apreciación, rojo deterioro.
      El número entre paréntesis es la cantidad de acciones del universo en ese sector.
      Permite identificar de un vistazo rotaciones y flujo de capital intra-día.
    </div>
  </section>

  <section class="card full-width">
    <div class="card-header"><span class="card-title">Distribución de Cambios y Market Breadth</span><span class="badge">universo completo</span></div>
    <div class="chart-wrap"><img src="data:image/png;base64,{img_breadth}" alt="Distribución y breadth"></div>
    <div class="explainer">
      <strong>Histograma (izquierda):</strong> Muestra cómo se distribuyen los cambios porcentuales
      del día. La línea punteada oscura es la media, la naranja la mediana, y las grises delimitan
      una desviación estándar. Un histograma concentrado indica movimientos homogéneos; colas anchas
      señalan alta dispersión y selectividad.<br><br>
      <strong>Market Breadth (derecha):</strong> Ratio Advance/Decline del universo completo.
      Un ratio mayor a 1.5 indica movimiento alcista amplio y participativo. Un ratio bajo con
      índices al alza puede señalar debilidad estructural.
    </div>
  </section>

  {volume_section}

  <section class="card full-width">
    <div class="card-header"><span class="card-title">Top 20 Ganadores y Perdedores — Barras por Sector</span><span class="badge">visualización</span></div>
    <div class="chart-wrap"><img src="data:image/png;base64,{img_movers}" alt="Movers chart"></div>
    <div class="explainer">
      <strong>Qué muestra:</strong> Representación gráfica de los top 20 ganadores y perdedores
      con el sector de pertenencia de cada instrumento. Permite detectar si los extremos del día
      están concentrados en uno o dos sectores (evento sectorial) o distribuidos en varios
      (movimiento amplio de mercado).
    </div>
  </section>

  <footer class="page-footer">
    Datos: TradingView Screener via tvscreener &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp;
    Solo con fines informativos, no constituye asesoramiento financiero.
  </footer>

</div>
</body>
</html>"""



with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"index.html generado correctamente — {now_str}")
