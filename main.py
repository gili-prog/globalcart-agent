from agent import run_agent

while True:
    customer_message = input("Enter customer message (or 'exit' to quit): ")
    if customer_message.lower() == 'exit':
        break
    run_agent(customer_message)