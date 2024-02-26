from cdm import CausalDecisionModel
from element import Lever, Outcome, Intermediate, External

lever1 = Lever("Marketing Spend", 500.00, [0.00, 2000.00])
lever2 = Lever("Sales Price", 8.25, [0.04, 15.00])
lever3 = Lever("Production Order Size", 85000, [0, 200000])

external1 = External("Market Size", 1000, [100, 10000])

itm1 = Intermediate("Demand", ["Sales Price"]).fit(lambda x: x)
itm2 = Intermediate("Uplift", ["Marketing Spend"]).fit(lambda x: x)
itm3 = Intermediate("Units Sold", ["Market Size", "Uplift", "Demand"]).fit(lambda x, y, z: x*y*z)
itm4 = Intermediate("Revenue", ["Units Sold", "Sales Price"]).fit(lambda x, y: x * y)

itm5 = Intermediate("Unit Cost", ["Production Order Size"]).fit(lambda x: x)
itm6 = Intermediate("Production Cost", ["Unit Cost", "Production Order Size"]).fit(lambda x, y: x * y)
itm7 = Intermediate("Total Cost", ["Marketing Spend", "Production Cost"]).fit(lambda x, y: x + y)

outcome = Outcome("Profit", ["Revenue", "Total Cost"]).fit(lambda x, y: x - y)

model = CausalDecisionModel([lever1, lever2, lever3])
model.add_element(external1)
model.add_elements([itm1, itm2, itm3, itm4, itm5, itm6, itm7])
model.add_element(outcome)

print(model.validate())
print(model.element("Total Cost"))
print(model.call("Profit")([1, 2, 3]))
print(model.sensitivity("Profit"))
x, y, z = model.multi_optimize(["Profit"])
print(x, y, z)
print(model.fidelity)

print(model.optimize("Profit"))
external1.value = 2000
print(model.call("Profit")([1, 2, 3]))
print(model.optimize("Profit"))