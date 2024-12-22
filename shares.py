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


if __name__ == '__main__':

    my_portfolio = 'COLBUN.SN AGUAS-A.SN HABITAT.SN'
    shares_count = my_portfolio.count(' ') + 1
    weights = np.array([1/shares_count] * shares_count)
    years = 5
    risk_free_rate = 0
    efficient_frontier = EfficientFrontier(my_portfolio, years, risk_free_rate, 100)
    print(efficient_frontier.efficient_frontier_keypoints_df())
    efficient_frontier.plot_efficient_portfolios()