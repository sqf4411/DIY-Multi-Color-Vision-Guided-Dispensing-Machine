import halcon as ha

print("Inspecting HDevProcedureCall...")
try:
    # We need a dummy procedure to create a call, or just inspect the class
    print(dir(ha.HDevProcedureCall))
except Exception as e:
    print(e)
