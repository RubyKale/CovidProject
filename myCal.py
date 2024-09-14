print("=====Welcome to Rohan Calculator====")
firstNumber = int(input("Enter num:"))
secondNumber = int(input("enter your second number:"))
sign = input("Enter one of the following "
             " \n + for addition,"
             "\n - for subtraction,"
             "\n / for division,"
             "\n * for multiplication \n")

print("First Number entered: " + str(firstNumber))
print("Second Number entered: " + str(secondNumber))
print("user selected to do: " + sign)
if sign == "+":
    print(firstNumber + secondNumber)
elif sign == "-":
    if firstNumber>secondNumber:
        print(firstNumber - secondNumber)
    else:
        print(secondNumber-firstNumber)
elif sign == "*":
    print(firstNumber * secondNumber)
elif sign == "/":
    print(firstNumber / secondNumber)
else:
    print("Operation not supported.")
