# Dashboard de Análisis de Acciones

Dashboard automatizado que monitorea el S&P 500 y genera análisis de mercado en tiempo real. Se actualiza automáticamente tres veces al día mediante GitHub Actions y se publica como página estática en GitHub Pages.

## Qué hace

Toma datos del screener de TradingView y calcula métricas de amplitud de mercado para detectar rotaciones sectoriales y movimientos significativos.

**Análisis incluidos:**

- Ratio Advance/Decline del universo completo
- Heatmap de sectores con cambio promedio del día
- Distribución de cambios porcentuales y desviación estándar
- Top 20 ganadores y perdedores
- Detección de volumen inusual (≥2x promedio 30 ruedas)
- Resumen ejecutivo generado desde los datos

## Stack técnico

- Python 3.11 con pandas, matplotlib y tvscreener
- GitHub Actions para ejecución programada
- GitHub Pages para hosting estático

## Cómo replicarlo

**1. Clonar o forkear este repositorio**

**2. Configurar GitHub Pages**

Settings → Pages → Source: Deploy from a branch → Branch: main → Folder: / (root) → Save

**3. Ejecutar el workflow manualmente la primera vez**

Actions → Actualizar Dashboard de Acciones → Run workflow

Esto genera el archivo index.html inicial. La URL pública va a ser:
```
https://TU_USUARIO.github.io/dashboard-stocks
```

**4. Ajustar horarios de actualización (opcional)**

Editar `.github/workflows/update_dashboard.yml` y modificar las líneas de cron:

```yaml
schedule:
  - cron: '0 14 * * 1-5'  # 11:00 AM hora Argentina
  - cron: '0 17 * * 1-5'  # 2:00 PM hora Argentina
  - cron: '0 21 * * 1-5'  # 6:00 PM hora Argentina
```

Los horarios están en UTC. Para convertir: hora Argentina + 3 horas = UTC.

## Estructura del proyecto

```
dashboard-stocks/
├── .github/
│   └── workflows/
│       └── update_dashboard.yml   # Configuración del cron job
├── generate_dashboard.py          # Script principal
├── requirements.txt               # Dependencias Python
└── index.html                     # Generado automáticamente
```

## Personalización

**Cambiar el universo de acciones:**

En `generate_dashboard.py`, modificar el índice después de crear el screener:

```python
ss = StockScreener()
ss.set_index(IndexSymbol.SP500)  # Cambiar por otro índice
```

Índices disponibles: SP500, NASDAQ_100, DOW_JONES, RUSSELL_2000, entre otros. Ver tvscreener docs.

**Agregar filtros adicionales:**

Después del select, antes del get:

```python
ss.select(...)
ss.filter(StockField.VOLUME > 1000000)  # Solo acciones con volumen > 1M
df = ss.get()
```

## Limitaciones conocidas

GitHub Actions puede demorar hasta 15 minutos en ejecutar los cron jobs programados. Si necesitás actualizaciones cada menos de 10 minutos, considerá un servidor dedicado.

GitHub suspende workflows sin actividad después de 60 días. Hacer cualquier commit al repo resetea el contador.

## Datos y descargo

Los datos provienen del screener público de TradingView via la librería tvscreener. El dashboard es solo con fines informativos y no constituye asesoramiento financiero.

## Licencia

MIT
