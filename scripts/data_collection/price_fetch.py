import requests
import csv
import time
import os


def get_crypto_to_usd_exchange_rates(ids):
    # CoinGecko API endpoint for multiple cryptocurrencies to USD exchange rates
    endpoint = "https://api.coingecko.com/api/v3/simple/price"

    # Parameters for the API request
    params = {
        "ids": ",".join(ids),  # Join the cryptocurrency IDs with a comma
        "vs_currencies": "usd"
    }

    try:
        response = requests.get(endpoint, params=params)
        data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def save_to_csv(timestamp, exchange_rates, filename, write_headers=False):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # if not file_exists or write_headers:
        # headers = ['timestamp'] + ids
        # writer.writerow(headers)

        row = [timestamp] + [exchange_rates[crypto_id]['usd']
                             for crypto_id in ids]
        writer.writerow(row)


if __name__ == "__main__":
    filename = "./data/crypto_exchange_rates.csv"
    ids = [
        "1inch",
        "aave",
        "arcblock",
        "alchemy-pay",
        "access-protocol",
        "cardano",
        "aergo",
        "adventure-gold",
        "aioz-network",
        "alchemix",
        "aleph-zero",
        "algorand",
        "my-neighbor-alice",
        "amp-token",
        "ankr",
        "aragon",
        "apecoin",
        "api3",
        "aptos",
        "arbitrum",
        "arpa",
        "as-monaco-fan-token",
        "astar",
        "automata",
        "cosmos",
        "auction",
        "audius",
        "aurora-near",
        "avalanche-2",
        "aventus",
        "axelar",
        "axie-infinity",
        "badger-dao",
        "balancer",
        "band-protocol",
        "basic-attention-token",
        "bitcoin-cash",
        "biconomy",
        "bitcoin",
        "blur",
        "bluzelle",
        "bancor",
        "barnbridge",
        "bitcoin",
        "braintrust",
        "binance-usd",
        "coin98",
        "coinbase-wrapped-staked-eth",
        "celer-network",
        "chiliz",
        "clover-finance",
        "internet-computer",
        "coti",
        "covalent",
        "crypto-com-chain",
        "crypterium",
        "curve-dao-token",
        "cartesi",
        "cortex",
        "civic",
        "convex-finance",
        "dai",
        "dash",
        "dash",
        "derivadao",
        "deso",
        "dextools",
        "dia-data",
        "dimo",
        "district0x",
        "dogecoin",
        "polkadot",
        "defi-yield-protocol",
        "elrond-erd-2",
        "elastos",
        "enjincoin",
        "ethereum-name-service",
        "eos",
        "ethernity-chain",
        "ethereum-classic",
        "ethereum",
        "eurocoinpay",
        "harvest-finance",
        "fetch-ai",
        "bonfida",
        "filecoin",
        "stafi",
        "flow",
        "flare-networks",
        "forta",
        "shapeshift-fox-token",
        "frax-share",
        "gala",
        "gala",
        "goldfinch",
        "aavegotchi",
        "moonbeam",
        "stepn",
        "gnosis",
        "gods-unchained",
        "the-graph",
        "green-satoshi-token",
        "gitcoin",
        "gemini-dollar",
        "gyen",
        "hedera-hashgraph",
        "hashflow",
        "high-performance-blockchain",
        "helium",
        "hopr",
        "internet-computer",
        "aurora-dao",
        "illuvium",
        "immutable-x",
        "frax-price-index-share",
        "injective-protocol",
        "inverse-finance",
        "iotex",
        "jasmycoin",
        "jupiter",
        "kava",
        "keep-network",
        "kyber-network-crystal",
        "kryll",
        "kusama",
        "lcx",
        "lido-dao",
        "chainlink",
        "litecoin",
        "league-of-kingdoms",
        "loom-network-new",
        "livepeer",
        "liquity",
        "loopring",
        "liquid-staked-ethereum",
        "litecoin",
        "magic",
        "decentraland",
        "mask-network",
        "math",
        "matic-network",
        "mcontent",
        "measurable-data-token",
        "media-licensing-token",
        "metis-token",
        "mina-protocol",
        "mirror-protocol",
        "maker",
        "melon",
        "marinade",
        "monavale",
        "maple",
        "msol",
        "metal",
        "elrond-erd-2",
        "musd",
        "muse-2",
        "mxc",
        "polyswarm",
        "near",
        "nest",
        "nkn",
        "numeraire",
        "nucypher",
        "ocean-protocol",
        "origin-protocol",
        "omisego",
        "ooki",
        "optimism",
        "orca",
        "orion-protocol",
        "osmosis",
        "orchid-protocol",
        "pax-gold",
        "perpetual-protocol",
        "playdapp",
        "pluton",
        "pangolin",
        "polkastarter",
        "matic-network",
        "marlin",
        "power-ledger",
        "echelon-prime",
        "near",
        "parsiq",
        "pundi-x-2",
        "vulcan-forged",
        "benqi",
        "quant-network",
        "quantstamp",
        "quickswap",
        "radix",
        "rainicorn",
        "superrare",
        "rarible",
        "ribbon-finance",
        "render-token",
        "republik",
        "request-network",
        "rari-governance-token",
        "iexec-rlc",
        "render-token",
        "oasis-network",
        "rocket-pool",
        "the-sandbox",
        "shiba-inu",
        "shping",
        "skale",
        "status",
        "havven",
        "rai-finance",
        "solana",
        "space-id",
        "spell-token",
        "stargate-finance",
        "storj",
        "blockstack",
        "sui",
        "suku",
        "superfarm",
        "sushi",
        "swftcoin",
        "sylo",
        "havven",
        "big-time",
        "te-food",
        "origintrail",
        "tellor",
        "tribe-2",
        "true-usd",
        "the-virtua-kolect",
        "uma",
        "unifi-protocol-dao",
        "uniswap",
        "pawtocol",
        "usd-coin",
        "tether",
        "ethos",
        "voxies",
        "wrapped-ampleforth",
        # "wrapped-axelar",
        "wrapped-bitcoin",
        "wrapped-centrifuge",
        "chain-2",
        "stellar",
        "xmon",
        "ripple",
        "tezos",
        "xyo-network",
        "yearn-finance",
        "yfii-finance",
        "zcash",
        "zencash",
        "0x"
    ]
    while True:
        timestamp = int(time.time())
        exchange_rates = get_crypto_to_usd_exchange_rates(ids)

        if exchange_rates is not None:
            print(f"Timestamp: {timestamp}, Exchange rates: {exchange_rates}")
            save_to_csv(timestamp, exchange_rates,
                        filename, write_headers=True)
        else:
            print("Failed to fetch the exchange rates.")

        # Wait for one minute before the next iteration
        time.sleep(60)
