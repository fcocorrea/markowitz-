# Modelo de Markowitz

## Introducción
Uno de los aspectos más relevantes en la vida es la capacidad de poder administrar nuestro dinero. Básicamente se trata de poder generar ahorros e invertir parte de nuestras ganancias.

Cuando logramos gastar menos de lo que ganamos y nos queda un sobrante, muchas veces nos podemos preguntar que hacemos. Una alternativa es invertir, pero comienzan a aparecer otras interrogantes como, donde lo invertimos, en qué activos y qué porcentaje de nuestro dinero le asignamos a cada uno de estos activos. En esta última interrogante me quiero enfocar y en brindar una solución.

Todos los activos tienen un riesgo asociado. Por ejemplo, las acciones están asociadas a un alto riesgo o volatilidad, pero hay acciones y acciones, como acciones de crecimiento, (Tesla o Meta) que son más volatiles y apuntan a un crecimiento rápido, o las acciones de dividendos que son mucho más estables (McDonald's o Coca Cola). Hay otros activos que tienen un riesgo menor a las acciones, como los bonos, pero de nuevo, hay bonos y bonos, como los bonos del tesoro de Estados Unidos, High Yield o los bonos corporativos, donde cada uno de ellos tiene asociado un riesgo distinto.

Además de poner atención al riesgo a la hora de invertir, debemos de poner atención a la diversificación. Es importante no poner todos los huevos en una misma canasta, es decir, debemos distribuir nuestro dinero en distintos activos con distintos riesgos e industrias. Por ejemplo, si ponemos todos nuestro dinero en acciones de la industria tecnológica no estamos diversificando de la mejor manera, porque si ocurre una noticia negativa para este sector todas nuestras acciones van a caer, pero si tengo acciones en tecnología, pero también en retail, puede que unas bajen, pero otras suban. De eso se trata la diversificación: protegerse ante subidas y bajadas del mercado, pudiendo tener mayor control sobre nuestro riesgo.

## Modelos de portafolios eficientes
Ante la inquietud de como podemos administrar nuestro riesgo de la mejor manera famosos economistas han dedicado sus estudios a resolver esta inquietud. Desde ahi han aparecido [modelos de portafolios de inversión](http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0123-77992011000100005) como el modelo de Black-Litterman o el modelo de Markowitz.

Este proyecto trata de automatizar el proceso de maximizar nuestra rentabilidad dado un riesgo asociado a nuestro portafolio de inversión. Para ello utilizaremos como referencia el modelo de Markowitz, que si bien tiene sus desventajas, puede ser de gran ayuda a la hora de decidir como vamos a distribuir nuestros activos de inversión para maximizar nuestro retorno dado un riesgo, o minimizar nuestro riesgo dado un retorno.

## Implementación

Si bien, el modelo se puede implementar en varias herramientas de manejo de datos, como en Excel con Solver, quise realizarlo en Python para utilizar la librería [yfinance](https://pypi.org/project/yfinance/) que se conecta a las APIs de Yahoo Finance para extraer información. Los pasos para construir este modelo son:

* Descargar los precios históricos de los activos financieros que nos interesan en un horizonte temporal de preferencia.
* Calcular los rendimientos históricos de los activos seleccionados y anualizamos el retorno multiplicando por 252 (se restan los fines de semana y feriados)
* Calculamos la matriz de covarianza para poder medir el riesgo del portafolio. Representa la relación entre las volatilidades de los activos.
* Definimos pesos aleatorios para cada activo dentro del portafolio. Por ejemplo, si tengo Pfizer, Tesla y Home Depot, podría poner un tercio para cada uno.
* Establecemos la restricción de este portafolio y es que los pesos deben sumar 1 (100%) y cada activo tendrá un peso que indica su proporción en el portafolio.
* Utilizando los pesos calculamos el rendimiento esperado y la desviación estándar (riesgo) del portafolio.
* Calculamos el ratio de Sharpe del portafolio, que mide la rentabilidad de una inversión ajustada al riesgo.
* Utilizamos un algoritmo de optimización para encontrar los pesos que minimicen la volatilidad o que maximicen el ratio de Sharpe.

De manera alternativa a los pasos anteriores, podemos también simular varios portafolios con pesos aleatorios hasta que nos encontremos con la **frontera eficiente** que corresponde al conjunto de carteras que ofrecen el mayor rendimiento posible para un nivel de riesgo determinado.

![frontera eficiente](fe.png)

Vemos que los puntos azules son varios portafolios aleatorios con distintos pesos, mientras que la linea anaranjada es donde están los portafolios más eficientes, es decir, aquellos que maximizan el retorno dado un riesgo.

## Código

Para este proyecto no incluí portafolios aleatorios. Simplemente calculé de inmediato la frontera eficiente y lo muestra en un gráfico. El proyecto también maximiza el íratio de Sharpe y minimiza el riesgo.

Para este proyecto utilizaremos la librería scipy de python que es ampliamente utilizada para problemas de optimización. El código completo es el siguiente:

```
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels import regression
import scipy.optimize as sc
import matplotlib.pyplot as plt


class Portfolio:
    def __init__(self, my_portfolio:str, years:int, risk_free_rate):
        # No todos estos atributos los utilizaré en todas las funciones. Ver que puedo sacar para una nueva clase y cuales se quedan.
        self.my_portfolio = my_portfolio
        self.years = years
        self.risk_free_rate = risk_free_rate
        self.asset_prices = self.historical_price()
        self.returns = self.asset_prices.pct_change()
        self.cov_matrix = self.returns.cov()
        self.mean_return = self.returns.mean()

    def date_range(self)-> tuple:
        """
        Rango de fechas donde se valoriza un activo desde el comienzo
        hasta el día de hoy. Por defecto, la clase ve un año de rango.
        El resultado es una tupla que contiene la fecha inicial y la fecha de hoy.

        retorna (fecha inicial, fecha de hoy)
        """
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=365 * self.years)
        return (start_date, today)

    def historical_price(self):
        """
        Precio histórico de un activo o activos.
        El output es un diccionario anidado, donde la clave es el activo y el 
        valor es otro diccionario cuya clave es el día y el valor es el precio.

        retorna {activo: {fecha:precio}}                
        """
        start_date, end_date = self.date_range()
        df_asset = yf.download(self.my_portfolio, start=start_date, end=end_date)
        header_levels = df_asset.columns.nlevels
        df_asset = df_asset[['Close']]
        if header_levels > 1: # dataframe con múltiples niveles
            df_asset = df_asset.droplevel(level=0, axis=1)
        else:
            df_asset = df_asset.rename(columns={'Close':self.symbol})
        return df_asset
    
    def portfolio_return(self, weights):
        """ Calculo del retorno anual del portafolio """
        portfolio_return = np.sum(self.mean_return * weights) * 252  # anualizamos
        return portfolio_return

    def portfolio_risk(self, weights):
        """ Calculo del riesgo del portafolio """
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights))) * np.sqrt(252)  # anualizamos
        return portfolio_risk
    
    def negative_sharpe_rate(self, weights):
        """ Calculo del sharpe rate negativo. """
        portfolio_return = self.portfolio_return(weights)
        portfolio_risk = self.portfolio_risk(weights)
        negative_sharpe_rate = - (portfolio_return - self.risk_free_rate) / portfolio_risk
        return negative_sharpe_rate

    def optimize(self, objective_function, target_return=None, bound=(0,1)):
        num_assets = len(self.mean_return)
        bounds = tuple(bound for asset in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if target_return != None:
            constraints.append({'type': 'eq', 'fun': lambda x:self.portfolio_return(x) - target_return})
        result = sc.minimize(objective_function, num_assets * [1/num_assets],
                             method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def plot_prices(self):
        plt.plot(self.asset_prices)
        plt.show()

    def calculate_beta(self, benchmark_symbol:str)->dict:
        """
        Calcular las pendientes para cada columna del DataFrame con respecto a la Serie.
        En el eje x está la variación de precios (retornos) del mercado, mientras que el eje y
        está la variación de precios de una empresa.
        """
        self.benchmark = Portfolio(benchmark_symbol, self.years) # CFMITNIPSA.SN puede ser un benchmark alternativo para el IPSA
        share_returns = self.returns[1:] # eliminamos nan
        betas = {}
        y = self.benchmark.returns.values[1:]
        for header, column in share_returns.items():
            x = column.values
            beta = self._beta_linreg(x,y)
            betas[header] = beta
        return pd.Series(betas)
    
    def _beta_linreg(self, x, y)->float:
        """ Calculamos los beta través de una 
        regresión lineal de dos activos financieros

        x: El mercado
        y: una acción
        """
        model = regression.linear_model.OLS(y,x).fit()
        beta = model.params[0]
        return beta

class EfficientFrontier:
    def __init__(self, my_portfolio, years, risk_free_rate, number_of_portfolios):        
        self.my_portfolio = my_portfolio
        self.years = years
        self.risk_free_rate = risk_free_rate
        self.number_of_portfolios = number_of_portfolios
        self.portfolio = Portfolio(my_portfolio, self.years, self.risk_free_rate)
        self.efficient_risks, self.efficient_returns, self.optimized_weights, \
            self.max_sharpe_rate_opt, self.min_risk_opt = self.efficient_portfolios()
        self.minRisk_return = self.portfolio.portfolio_return(self.min_risk_opt.x)
        self.minRisk_risk = self.portfolio.portfolio_risk(self.min_risk_opt.x)
        self.maxSR_return = self.portfolio.portfolio_return(self.max_sharpe_rate_opt.x)
        self.maxSR_risk = self.portfolio.portfolio_risk(self.max_sharpe_rate_opt.x)

    def efficient_portfolios(self):
        max_sharpe_rate_opt = self.portfolio.optimize(self.portfolio.negative_sharpe_rate)
        min_risk_opt = self.portfolio.optimize(self.portfolio.portfolio_risk)
        target_returns = np.linspace(min_risk_opt.fun, max_sharpe_rate_opt.fun, self.number_of_portfolios)
        efficient_risks, efficient_returns, efficient_weights = [], [], []
        for target in target_returns:
            optimized_portfolio= self.portfolio.optimize(self.portfolio.portfolio_risk, target)
            optimized_weights = optimized_portfolio.x
            optimized_portfolio_risk = self.portfolio.portfolio_risk(optimized_weights)
            optimized_portfolio_return = self.portfolio.portfolio_return(optimized_weights)
            efficient_risks.append(optimized_portfolio_risk)
            efficient_returns.append(optimized_portfolio_return)
            efficient_weights.append({asset_name:weight for asset_name, weight in zip(self.my_portfolio.split(), optimized_weights)})
        return efficient_risks, efficient_returns, efficient_weights, max_sharpe_rate_opt, min_risk_opt
    
    def efficient_portfolio_df(self):
        weights_list = [list(weights.values()) for weights in self.optimized_weights]
        returns = [self.portfolio.portfolio_return(weights) for weights in weights_list]
        risks = [self.portfolio.portfolio_risk(weights) for weights in weights_list]
        df = pd.DataFrame(self.optimized_weights).reset_index()
        df['Retorno'] = returns
        df['Riesgo'] = risks
        df['Sharpe Rate'] = (df['Retorno'] - self.portfolio.risk_free_rate) / df['Riesgo']
        df = df.mul(100).round(2).drop_duplicates()
        return df
    
    def efficient_frontier_keypoints_df(self):
        max_sharpe_rate_row = list(self.max_sharpe_rate_opt.x) + [self.maxSR_return, self.maxSR_risk]
        min_risk_row = list(self.min_risk_opt.x) + [self.minRisk_return, self.minRisk_risk]
        columns = self.my_portfolio.split() + ['Retorno', 'Riesgo']
        df = pd.DataFrame([max_sharpe_rate_row, min_risk_row], columns=columns)
        df = df.mul(100).round(2).drop_duplicates()
        df['Optimización'] = ['Max Sharpe', 'Min Riesgo']
        return df
    
    def plot_efficient_portfolios(self):
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))        
        plt.scatter(x=self.efficient_risks, y=self.efficient_returns, c='orange', s=10, label='Portafolio')        
        self.plot_efficient_frontier(self.efficient_risks, self.efficient_returns, self.minRisk_return)
        self.plot_assets()
        plt.scatter(self.minRisk_risk, self.minRisk_return, color='green', marker='*', s=200, label='Min Riesgo')
        plt.scatter(self.maxSR_risk, self.maxSR_return, color='red', marker='*', s=200, label='Max Sharpe')
        plt.axis('equal')
        plt.title("Frontera Eficiente")
        plt.xlabel("Volatilidad anualizada")
        plt.ylabel("Rendimiento anualizado")
        plt.legend(loc='lower right')
        plt.show()

    def plot_efficient_frontier(self, efficient_risks, efficient_returns, minRisk_return):
        efficient_frontier = [(risk, profit) for risk, profit in zip(efficient_risks, efficient_returns)
                              if profit >= minRisk_return]
        plt.plot(*zip(*efficient_frontier), linestyle='dashed', color='black', label='Frontera Eficiente',
                 linewidth=3.0)
        
    def plot_assets(self):
        asset_prices = self.portfolio.asset_prices
        asset_prices['index'] = range(asset_prices.shape[0])
        years_to_days = asset_prices.shape[0] // self.years
        asset_prices = asset_prices[asset_prices['index'] % years_to_days == 0]
        returns = asset_prices.pct_change()
        mean_return = returns.mean()
        std_return = returns.std()
        for asset in asset_prices.columns:
            plt.scatter(std_return[asset], mean_return[asset], color='blue')
            plt.annotate(asset, (std_return[asset], mean_return[asset]), textcoords="offset points", xytext=(0,10), ha='center')
```

El modo de uso de este código es simple: debemos escoger los activos que queremos analizar, buscamos sus nemotécnicos y los instanciamos en la clase EfficientFrontier separando cada nemotécnico con un espacio. A su vez, elegimos el horizonte temporal al que queremos realizar el análisis.

Por ejemplo, imaginemos que queremos calcular la frontera eficiente de tres acciones chilenas que pertenecen a distintas industrias: Colbun, Aguas Andinas y AFP Habitat.

Buscamos sus nemotécnicos en Yahoo Finance y tenemos lo siguiente:
* Nemotécnico de Colbun: COLBUN.SN
* Nemotécnico de Aguas Andinas: AGUAS-A.SN
* Nemotécnico de AFP Habitat: HABITAT.SN

Ahora, supongamos que quiero hacer el análisis de estas 3 acciones en un plazo de 5 años.

Ya con esos datos soy capaz de calcular la frontera eficiente. En código se ve así:

```
if __name__ == '__main__':

    my_portfolio = 'BTC-USD ETH-USD SOL-USD UNI7083-USD'
    shares_count = my_portfolio.count(' ') + 1
    weights = np.array([1/shares_count] * shares_count)
    years = 5
    risk_free_rate = 0
    efficient_frontier = EfficientFrontier(my_portfolio, years, risk_free_rate, 100)
    print(efficient_frontier.efficient_frontier_keypoints_df())
    efficient_frontier.plot_efficient_portfolios()
```
Definimos una tasa libre de riesgo de cero, los pesos los dividimos en tres partes iguales e instanciamos todo en la clase EfficientFrontier para mostrar la frontera eficiente. Finalmente imprimimos una tabla donde definimos los pesos para cada activo cuando minimizamos riesgo y cuando maximizamos el ratio de Sharpe. Además, generamos el gráfico de la frontera eficiente.

![gráfico de frontera eficiente](resultado.png)

La estrella verde del gráfico indica el portafolio con el minimo riesgo posible que máxima la rentabilidad y se encuentra al comienzo de la frontera eficiente. La estrella roja, en cambio, indica el portafolio con el máximo ratio de Sharpe posible. Vemos también el retorno y riesgo asociado a cada acción durante 5 años, marcadas con puntos azules.

Los pesos para cada acción cuando minimizamos el riesgo y maximizamos el ratio de Sharpe es el siguiente:

| COLBUN.SN | AGUAS-A.SN | HABITAT.SN | Retorno | Riesgo | Optimización  |
|-----------|------------|------------|---------|--------|---------------|
| 0.00      | 52.18      | 47.82      | 5.30    | 27.64  | Max Sharpe    |
| 30.74     | 21.40      | 47.86      | 3.65    | 24.24  | Min Riesgo    |

En un escenario donde queremos maximizar el ratio Sharpe no invertimos nada en Colbun, si no que nos distribuimos entre Aguas Andinas y AFP Habitat, pero si estamos en un escenario donde queremos minimizar el riesgo, si lo incluimos.

De la misma forma puedes intentarlo con cualquier activo, sin importar si es una acción, bono, ETF, criptomoneda, etc. Lo genial de esto es que puedes mezclar muchos tipos de activos en tu portafolio y siempre llegar a la distribución más optima.









