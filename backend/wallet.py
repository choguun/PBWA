from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from web3 import Web3
from web3.middleware import SignAndSendRawMiddlewareBuilder
from eth_account.signers.local import LocalAccount
from eth_account import Account

from goat_adapters.langchain import get_on_chain_tools
from goat_plugins.erc20.token import PEPE, USDC
from goat_plugins.erc20 import erc20, ERC20PluginOptions
from goat_wallets.evm import send_eth
from goat_wallets.web3 import Web3EVMWalletClient