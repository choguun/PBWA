from web3 import Web3
from web3.middleware import SignAndSendRawMiddlewareBuilder
from eth_account.signers.local import LocalAccount
from eth_account import Account

from goat_wallets.web3 import Web3EVMWalletClient

# Import configuration
from .config import WALLET_PRIVATE_KEY, RPC_PROVIDER_URL

class EVMWallet:
    def __init__(self):
        """Initializes the EVM wallet using configuration settings."""
        # Validate configuration
        if not RPC_PROVIDER_URL:
            raise ValueError("RPC_PROVIDER_URL environment variable not set!")
        if not WALLET_PRIVATE_KEY:
            raise ValueError("WALLET_PRIVATE_KEY environment variable not set!")
        if not WALLET_PRIVATE_KEY.startswith("0x"):
            raise ValueError("Private key must start with 0x hex prefix")

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(RPC_PROVIDER_URL))

        # Initialize account
        self.account: LocalAccount = Account.from_key(WALLET_PRIVATE_KEY)

        # Set default account and add middleware
        self.w3.eth.default_account = self.account.address
        self.w3.middleware_onion.add(
            SignAndSendRawMiddlewareBuilder.build(self.account)
        )

        # Initialize Goat SDK Wallet Client
        self.client = Web3EVMWalletClient(self.w3)

    def get_client(self) -> Web3EVMWalletClient:
        """Returns the initialized Web3EVMWalletClient."""
        return self.client

    def get_web3_instance(self) -> Web3:
        """Returns the initialized Web3 instance."""
        return self.w3

    def get_account(self) -> LocalAccount:
        """Returns the initialized LocalAccount."""
        return self.account

