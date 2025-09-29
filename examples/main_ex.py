import pandas as pd
import json
from src.function_calling import FunctionCalling

import asyncio

# Assuming we have the following data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}
df = pd.DataFrame(data)

def retrieve_payment_status(transaction_id: str) -> str:
    """
    Get payment status of a transaction.
    :param transaction_id: The transaction id.
    """
    if transaction_id in df['transaction_id'].values:
        status = df[df['transaction_id'] == transaction_id]['payment_status'].item()
        return json.dumps({'status': status})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(transaction_id: str) -> str:
    """
    Get payment date of a transaction.
    :param transaction_id: The transaction id.
    """
    if transaction_id in df['transaction_id'].values:
        date = df[df['transaction_id'] == transaction_id]['payment_date'].item()
        return json.dumps({'date': date})
    return json.dumps({'error': 'transaction id not found.'})

async def main():
    # Initialize the FunctionCalling handler
    handler = FunctionCalling(model="qwen/qwen3-30b-a3b-thinking-2507")

    # Register the tools
    handler.register_tool(retrieve_payment_status)
    handler.register_tool(retrieve_payment_date)
    
    # Run the function calling process with a user prompt
    user_prompt = "What's the status of my transaction T1001?"
    final_answer = await handler.run_async(user_prompt)
    print(final_answer)

    user_prompt_date = "When was transaction T1003 paid?"
    final_answer_date = await handler.run_async(user_prompt_date)
    print(final_answer_date)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())