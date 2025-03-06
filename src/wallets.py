from os.path import expanduser

from bittensor_wallet.wallet import Wallet


# ✏️ Edit the wallet names to match the names you see when using `btcli wallet list`
def get_miner_wallet() -> Wallet:  # type: ignore
    return Wallet(name="your_coldkey_name", hotkey="hotkey_name_name")


def get_miner_wallet_hotkey_path(wallet: Wallet) -> str:  # type: ignore
    return expanduser(f"~/.bittensor/wallets/{wallet.name}/hotkeys/{wallet.hotkey_str}")


if __name__ == "__main__":
    wallet = get_miner_wallet()
    print(wallet)

    hotkey_path = get_miner_wallet_hotkey_path(wallet)
    print(f"Hotkey path: {hotkey_path}")
