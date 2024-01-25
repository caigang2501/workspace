from QuantLib import *

# 定义日期和计息期
calendar = TARGET()
settlement_date = Date(15, 5, 2023)
maturity_date = Date(15, 5, 2024)

# 构建欧式期权
option_type = Option.Call
underlying_price = 100.0
strike_price = 105.0
option_start_date = settlement_date
option_end_date = maturity_date
day_counter = Actual360()
exercise_type = EuropeanExercise

payoff = PlainVanillaPayoff(option_type, strike_price)
exercise = EuropeanExercise(option_end_date)

european_option = VanillaOption(payoff, exercise)

# 构建市场场景
underlying = SimpleQuote(underlying_price)
flat_term_structure = FlatForward(settlement_date, QuoteHandle(SimpleQuote(0.01)), day_counter)
flat_vol_term_structure = BlackConstantVol(settlement_date, calendar, QuoteHandle(SimpleQuote(0.20)), day_counter)

bsm_process = BlackScholesProcess(QuoteHandle(underlying),
                                  YieldTermStructureHandle(flat_term_structure),
                                  BlackVolTermStructureHandle(flat_vol_term_structure))

european_option.setPricingEngine(AnalyticEuropeanEngine(bsm_process))

# 计算期权价格
option_price = european_option.NPV()

# 输出结果
print(f"欧式期权价格: {option_price}")
