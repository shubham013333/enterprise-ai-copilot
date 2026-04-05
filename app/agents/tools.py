def calculator_tool(query: str):
    try:
        return str(eval(query))
    except:
        return "Invalid calculation"