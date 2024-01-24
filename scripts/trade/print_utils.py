from prettytable import PrettyTable

def portfolio_table(portfolio):
    total_usdc = 0
    table = PrettyTable(["Currency", "Balance (Coin)",
                        "Balance (USD)", "Allocation"])
    table.align["Currency"] = "l"
    table.align["Balance (Coin)"] = "r"
    table.align["Balance (USD)"] = "r"
    table.align["Allocation"] = "r"
    for i, holding in enumerate(portfolio):
        total_usdc += holding.balance_usd

        last_row = (i == len(portfolio) - 1)
        table.add_row([
            holding.currency,
            holding.balance_coin,
            "{:.2f}".format(holding.balance_usd),
            "{:.2f}%".format(holding.allocation * 100.0)
        ], divider=last_row)

    table.add_row([
        "TOTAL",
        "USDC",
        "{:.2f}".format(total_usdc),
        "-"
    ])
    return table
