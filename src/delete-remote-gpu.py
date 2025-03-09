import asyncio
import datetime
import hashlib
import json
import time
import argparse
from typing import Any, Dict, TypedDict

import aiohttp
from bittensor_wallet.wallet import Wallet
from rich import box
from rich.console import Console
from rich.table import Table
from substrateinterface import Keypair

from wallets import get_miner_wallet, get_miner_wallet_hotkey_path

VALIDATOR_HEADER = "X-Chutes-Validator"
MINER_HEADER = "X-Chutes-Miner"
NONCE_HEADER = "X-Chutes-Nonce"
SIGNATURE_HEADER = "X-Chutes-Signature"
HOTKEY_HEADER = "X-Chutes-Hotkey"

CHUTES_API = "https://api.chutes.ai"


class RemoteInventoryItem(TypedDict):
    uuid: str
    name: str
    memory: int
    major: int
    minor: int
    processors: int
    sxm: bool
    clock_rate: float
    max_threads_per_processor: int
    concurrent_kernels: bool
    ecc: bool
    seed: int
    miner_hotkey: str
    gpu_identifier: str
    device_index: int
    created_at: str
    verification_host: str
    verification_port: int
    verification_error: str
    verified_at: str
    chute: str
    chute_id: str
    inst_verification_error: str
    inst_verified_at: str


def scorch_remote(miner_wallet: Wallet, auto_delete: bool):  # type: ignore
    remote_inventory: list[RemoteInventoryItem] = fetch_remote_inventory(miner_wallet)
    hotkey_path = get_miner_wallet_hotkey_path(miner_wallet)
    validator_api = CHUTES_API

    async def _scorch_remote():
        nonlocal remote_inventory, hotkey_path, validator_api

        ip_to_gpus = {}

        for gpu in remote_inventory:
            gpu_ip = gpu["verification_host"]
            if gpu_ip not in ip_to_gpus:
                ip_to_gpus[gpu_ip] = []
            ip_to_gpus[gpu_ip].append(gpu)

        # Display remote inventory so user can choose what to delete
        display_gpu_table(remote_inventory, miner_wallet, title="Remote Inventory")

        print("⚠️ CAUTION, it'll remove the gpu from the remote inventory.\n")

        if auto_delete:
            selected_gpus = [inventory for inventory in remote_inventory if inventory['chute'] is None and inventory['chute_id'] is None]
            display_gpu_table(selected_gpus, miner_wallet, title="Selected GPUs for Deletion")
        else:
            selected_gpus = prompt_user_input(remote_inventory, ip_to_gpus)
            display_gpu_table(selected_gpus, miner_wallet, title="Selected GPUs for Deletion")

            # Display confirmation table before deletion
            input("🚨 Press Enter to confirm deletion...")

        if not selected_gpus:
            return

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            for gpu in selected_gpus:
                headers, _ = sign_request(hotkey_path, purpose="nodes", remote=True)
                name = gpu["name"]
                uuid = gpu["uuid"]
                try:
                    async with session.delete(f"{validator_api.rstrip('/')}/nodes/{uuid}", headers=headers) as resp:
                        print(f"✅ Successfully deleted {name} (UUID {uuid}).")
                        time.sleep(0.25)
                except aiohttp.ClientError as e:
                    print(f"❌ Failed to delete {name} (UUID {uuid}): {e}")

    asyncio.run(_scorch_remote())


def prompt_user_input(remote_inventory: list[RemoteInventoryItem], ip_to_gpus: dict[str, list[RemoteInventoryItem]]) -> list[RemoteInventoryItem] | None:
    """
    Prompts user for GPU deletion input.
    Supports:
    - Entering multiple indexes (comma/space separated).
    - Entering an IP to delete all GPUs on that IP.

    Returns:
    - A list of GPUs to delete.
    - None if input is invalid.
    """
    user_input = input("Enter the indexes of GPUs to remove (separated by spaces or commas) or enter an IP: ").strip()

    # Check if input is an IP (basic validation)
    if user_input.count(".") == 3 and all(part.isdigit() and 0 <= int(part) <= 255 for part in user_input.split(".")):
        if user_input in ip_to_gpus:
            selected_gpus = ip_to_gpus[user_input]
            print(f"🔍 Found {len(selected_gpus)} GPUs on IP {user_input}. Proceeding with deletion...")
            return selected_gpus
        else:
            print(f"❌ No GPUs found on IP {user_input}. Aborting.")
            return None
    else:
        # Process as index input
        try:
            indexes = [int(idx) for idx in user_input.replace(",", " ").split() if idx.strip().isdigit()]
            selected_gpus = [remote_inventory[i] for i in indexes if i < len(remote_inventory)]

            if not selected_gpus:
                print("❌ No valid GPUs selected. Aborting.")
                return None
            return selected_gpus
        except (ValueError, IndexError):
            print("❌ Invalid input format. Please enter numbers or a valid IP.")
            return None


def get_signing_message(
    hotkey: str,
    nonce: str,
    payload_str: str | bytes | None,
    purpose: str | None = None,
    payload_hash: str | None = None,
) -> str:
    """
    Get the signing message for a given hotkey, nonce, and payload.
    """
    if payload_str:
        if isinstance(payload_str, str):
            payload_str = payload_str.encode()
        return f"{hotkey}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{hotkey}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{hotkey}:{nonce}:{payload_hash}"
    else:
        raise ValueError("Either payload_str or purpose must be provided")


def sign_request(
    hotkey: str,
    payload: Dict[str, Any] | str | None = None,
    purpose: str = None,
    remote: bool = False,
):
    """
    Generate a signed request (for miner requests to validators).
    """
    hotkey_data = json.loads(open(hotkey).read())
    nonce = str(int(time.time()))
    headers = {
        MINER_HEADER: hotkey_data["ss58Address"],
        NONCE_HEADER: nonce,
    }
    if remote:
        headers[HOTKEY_HEADER] = headers.pop(MINER_HEADER)
    signature_string = None
    payload_string = None
    if payload is not None:
        if isinstance(payload, (list, dict)):
            headers["Content-Type"] = "application/json"
            payload_string = json.dumps(payload)
        else:
            payload_string = payload
        signature_string = get_signing_message(
            hotkey_data["ss58Address"],
            nonce,
            payload_str=payload_string,
            purpose=None,
        )
    else:
        signature_string = get_signing_message(hotkey_data["ss58Address"], nonce, payload_str=None, purpose=purpose)
    if not remote:
        signature_string = hotkey_data["ss58Address"] + ":" + signature_string
        headers[VALIDATOR_HEADER] = headers[MINER_HEADER]
    keypair = Keypair.create_from_seed(hotkey_data["secretSeed"])
    headers[SIGNATURE_HEADER] = keypair.sign(signature_string.encode()).hex()
    return headers, payload_string


def format_memory(memory_bytes):
    """
    Convert memory from bytes to GB and format nicely.
    """
    return f"{memory_bytes / (1024**3):.1f}GB"


def format_date(date_str):
    """
    Format datetime string to a more readable format.
    """
    dt = datetime.datetime.fromisoformat(date_str)
    return dt.strftime("%Y-%m-%d %H:%M")


def format_verification(error, verified_at):
    """
    Helper to format table cell for GPU verification.
    """
    if verified_at:
        return f"[green]Verified: {format_date(verified_at)}[/green]"
    elif error:
        return f"[red]Error: {error}[/red]"
    return "[yellow]Pending[/yellow]"


def display_remote_inventory(inventory):
    """
    Render remote/validator inventory.
    """
    console = Console()
    table = Table(title="GPU Information")
    table.add_column("Name", style="cyan")
    table.add_column("Chute", style="cyan")
    table.add_column("Memory (GB)", justify="right", style="green")
    table.add_column("Clock (MHz)", justify="right", style="red")
    table.add_column("Created At", style="blue")
    table.add_column("GPU Verification", style="white")
    table.add_column("Instance verification", style="white")
    for gpu in inventory:
        table.add_row(
            gpu["name"],
            f"{gpu['chute_id']} {gpu['chute']}",
            format_memory(gpu["memory"]),
            f"{gpu['clock_rate'] / 1000:.0f}",
            format_date(gpu["created_at"]),
            format_verification(gpu["verification_error"], gpu["verified_at"]),
            format_verification(gpu["inst_verification_error"], gpu["inst_verified_at"]),
        )
    console.print(table)
    console.print("\n" + "=" * 80 + "\n")


def fetch_remote_inventory(miner_wallet: Wallet) -> list[RemoteInventoryItem]:  # type: ignore
    """
    GET remote inventory for hotkey (i.e., what the validator has tracked) inventory.
    [
        {
            "uuid": "bbf86277cb61e748ef875baa27b5f0ed",
            "name": "NVIDIA L40S",
            "memory": 47697362944,
            "major": 8,
            "minor": 9,
            "processors": 142,
            "sxm": false,
            "clock_rate": 2520000.0,
            "max_threads_per_processor": 1536,
            "concurrent_kernels": true,
            "ecc": true,
            "seed": 9095026960248786810,
            "miner_hotkey": "hotkey-ss58",
            "gpu_identifier": "l40s",
            "device_index": 0,
            "created_at": "2025-01-31T07:56:59.673308+00:00",
            "verification_host": "160.202.129.197",
            "verification_port": 31285,
            "verification_error": null,
            "verified_at": "2025-01-31T07:57:07.657239+00:00",
            "chute": "Qwen/Qwen2.5-72B-Instruct",
            "chute_id": "62cc0462-8983-5ef1-8859-92ccf726e235",
            "inst_verification_error": null,
            "inst_verified_at": "2025-02-01T17:50:36.081287+00:00"
        }
    ]
    """
    hotkey_path = get_miner_wallet_hotkey_path(miner_wallet)
    validator_api = CHUTES_API

    async def _remote_inventory():
        nonlocal hotkey_path, validator_api
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers, _ = sign_request(hotkey_path, purpose="miner", remote=True)
            inventory = []
            gpu_map = {}
            async with session.get(f"{validator_api}/miner/nodes/", headers=headers) as resp:
                async for content_enc in resp.content:
                    content = content_enc.decode()
                    if content.startswith("data: "):
                        inventory.append(json.loads(content[6:]))
                        gpu_map[inventory[-1]["uuid"]] = inventory[-1]
                        gpu_map[inventory[-1]["uuid"]].update(
                            {
                                "chute": None,
                                "chute_id": None,
                                "inst_verification_error": None,
                                "inst_verified_at": None,
                            }
                        )
            async with session.get(f"{validator_api}/miner/inventory", headers=headers) as resp:
                for item in await resp.json():
                    if item["gpu_id"] in gpu_map:
                        gpu_map[item["gpu_id"]].update(
                            {
                                "chute": item["chute_name"],
                                "chute_id": item["chute_id"],
                                "inst_verification_error": item["verification_error"],
                                "inst_verified_at": item["last_verified_at"],
                            }
                        )
            inventory = sorted(inventory, key=lambda o: o["created_at"])
            return inventory

    return asyncio.run(_remote_inventory())


def display_gpu_table(remote_inventory: list[RemoteInventoryItem], miner_wallet: Wallet, title="Remote Inventory"):  # type: ignore
    console = Console()

    server_table = Table(title=f"{title}: {miner_wallet}", box=box.ROUNDED)

    server_table.add_column("#", style="white")
    server_table.add_column("GPU Name", style="blue")
    server_table.add_column("GPU UUID", style="magenta")
    server_table.add_column("Created At", style="green")
    server_table.add_column("Server IP", style="red")
    server_table.add_column("Chute", style="yellow")

    for i, gpu in enumerate(remote_inventory):
        server_table.add_row(str(i), gpu["name"], gpu["uuid"], gpu["created_at"], gpu["verification_host"], gpu["chute"])

    console.print(server_table)
    console.print()

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--auto-delete', action='store_true', help="Delete gpu_id automatically.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args: argparse.Namespace = get_cli_args()
    auto_delete: bool = args.auto_delete

    print("☠️ Scorch Remote Inventory...")

    wallet: Wallet = get_miner_wallet()  # type: ignore
    if not wallet:
        raise ValueError("Failed to get miner wallet")

    scorch_remote(miner_wallet=wallet, auto_delete=auto_delete)
