from prettytable import PrettyTable

def portfolio_table(portfolio):
    table = PrettyTable(["Currency", "Balance (Coin)",
                        "Balance (USD)", "Allocation"])
    table.align["Currency"] = "l"
    table.align["Balance (Coin)"] = "r"
    table.align["Balance (USD)"] = "r"
    table.align["Allocation"] = "r"
    for holding in portfolio:
        table.add_row([
            holding.currency,
            holding.balance_coin,
            "{:.2f}".format(holding.balance_usd),
            "{:.2f}%".format(holding.allocation * 100.0)
        ])
    return table
