from agent import run_agent

while True:
    customer_message = input("Enter customer message (or 'exit' to quit): ")
    if customer_message.lower() == 'exit':
        break
    try:
        result = run_agent(customer_message)
        print(result["final_response"].customer_response)
    except Exception as e:
        print(f"Error occurred: {e}")
    
    