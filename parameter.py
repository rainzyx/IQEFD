from model import convnext_small as create_model
from modelconvnext import convnext_base as create_model1
model=create_model(3)
model1=create_model1(3)
print(model)
print(model1)
parameter = sum([param.nelement() for param in model.parameters()])
parameter1 = sum([param.nelement() for param in model1.parameters()])
print("Model number of parameter: %.2fM" % (parameter/1e6))
print("Model1 number of parameter: %.2fM" % (parameter1/1e6))
total = parameter + parameter1
print("Total number of parameter: %.2fM" % (total/1e6))