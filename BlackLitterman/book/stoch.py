import QuantLib as ql

spot = 58.0
strike = 55
T = ql.Period(8, ql.Days)
r = 0.02

# Market data
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()
todays_date = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = todays_date

# Heston process
v0 = 0.04
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.5

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
r_handle = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, r, day_count))
div_handle = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, 0.0, day_count))

heston_process = ql.HestonProcess(r_handle, div_handle, spot_handle, v0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)
engine = ql.AnalyticHestonEngine(heston_model)

# Option
expiry = todays_date + T
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
european_exercise = ql.EuropeanExercise(expiry)
option = ql.VanillaOption(payoff, european_exercise)
option.setPricingEngine(engine)

print("Heston price:", option.NPV())
