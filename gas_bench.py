"""
Measures the gas cost of the on-chain integrity layer of blockchain-based federated learning.

The contract (FLLedger.sol, included below as source and as compiled bytecode) lets each client
commit the SHA-256 hash of its model update, lets the aggregator or committee record accept/reject
decisions that update an on-chain reputation, and commits the hash of the new global model.

It runs on a local in-memory Ethereum chain (eth-tester with py-evm), so no network, wallet or
real ether is needed. Gas is deterministic, so the numbers do not depend on the machine.

Usage:
  python gas_bench.py                     # 50 clients, 3 rounds
  python gas_bench.py --clients 10 50 100 --rounds 3
"""
import argparse
import json
import csv
import hashlib
import statistics

from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend

SOLC_VERSION = '0.8.24+commit.e11b9ed9.Emscripten.clang'
ABI = json.loads(r"""[{"inputs": [], "stateMutability": "nonpayable", "type": "constructor"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "uint256", "name": "round", "type": "uint256"}, {"indexed": true, "internalType": "address", "name": "client", "type": "address"}, {"indexed": false, "internalType": "bool", "name": "accepted", "type": "bool"}], "name": "Decision", "type": "event"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "uint256", "name": "round", "type": "uint256"}, {"indexed": false, "internalType": "bytes32", "name": "hash", "type": "bytes32"}], "name": "GlobalCommitted", "type": "event"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "uint256", "name": "round", "type": "uint256"}, {"indexed": true, "internalType": "address", "name": "client", "type": "address"}, {"indexed": false, "internalType": "bytes32", "name": "hash", "type": "bytes32"}], "name": "UpdateSubmitted", "type": "event"}, {"inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}, {"internalType": "bytes32", "name": "h", "type": "bytes32"}], "name": "commitGlobal", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "globalHash", "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}], "stateMutability": "view", "type": "function"}, {"inputs": [], "name": "owner", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}, {"internalType": "address", "name": "c", "type": "address"}, {"internalType": "bool", "name": "ok", "type": "bool"}], "name": "recordDecision", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}, {"internalType": "address[]", "name": "cs", "type": "address[]"}, {"internalType": "bool[]", "name": "oks", "type": "bool[]"}], "name": "recordDecisions", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address[]", "name": "clients", "type": "address[]"}], "name": "registerClients", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "", "type": "address"}], "name": "registered", "outputs": [{"internalType": "bool", "name": "", "type": "bool"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "address", "name": "", "type": "address"}], "name": "reputation", "outputs": [{"internalType": "uint16", "name": "", "type": "uint16"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}, {"internalType": "bytes32", "name": "h", "type": "bytes32"}], "name": "submitUpdate", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}, {"internalType": "address", "name": "", "type": "address"}], "name": "updateHash", "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}], "stateMutability": "view", "type": "function"}]""")
BYTECODE = '0x608060405234801561001057600080fd5b50600080546001600160a01b03191633179055610a02806100326000396000f3fe608060405234801561001057600080fd5b506004361061009e5760003560e01c80638da5cb5b116100665780638da5cb5b1461012f578063b2dd5c071461015a578063b9f794511461018d578063c4360a8f146101c4578063cd4b8961146101e457600080fd5b80631b002fea146100a35780631e34d0cb146100b85780633931cd3f146100cb5780633c267887146100de5780636a2e2512146100f1575b600080fd5b6100b66100b13660046106f1565b6101f7565b005b6100b66100c636600461075f565b6102cb565b6100b66100d93660046107a1565b6103cb565b6100b66100ec3660046106f1565b61049f565b61011c6100ff366004610837565b600360209081526000928352604080842090915290825290205481565b6040519081526020015b60405180910390f35b600054610142906001600160a01b031681565b6040516001600160a01b039091168152602001610126565b61017d610168366004610863565b60016020526000908152604090205460ff1681565b6040519015158152602001610126565b6101b161019b366004610863565b60026020526000908152604090205461ffff1681565b60405161ffff9091168152602001610126565b61011c6101d2366004610885565b60046020526000908152604090205481565b6100b66101f23660046108ae565b61059e565b6000546001600160a01b0316331461022a5760405162461bcd60e51b8152600401610221906108ea565b60405180910390fd5b6000828152600460205260409020541561027a5760405162461bcd60e51b8152602060048201526011602482015270185b1c9958591e4818dbdb5b5a5d1d1959607a1b6044820152606401610221565b600082815260046020526040908190208290555182907f5c0832e1cf2ed1266292f33637ff24d8e878fdaf5b2ed13d6065aaa210cceddc906102bf9084815260200190565b60405180910390a25050565b6000546001600160a01b031633146102f55760405162461bcd60e51b8152600401610221906108ea565b60005b818110156103c65760018060008585858181106103175761031761090d565b905060200201602081019061032c9190610863565b6001600160a01b03166001600160a01b0316815260200190815260200160002060006101000a81548160ff0219169083151502179055506127106002600085858581811061037c5761037c61090d565b90506020020160208101906103919190610863565b6001600160a01b031681526020810191909152604001600020805461ffff191661ffff929092169190911790556001016102f8565b505050565b6000546001600160a01b031633146103f55760405162461bcd60e51b8152600401610221906108ea565b82811461042d5760405162461bcd60e51b81526020600482015260066024820152650d8cadccee8d60d31b6044820152606401610221565b60005b838110156104975761048f8686868481811061044e5761044e61090d565b90506020020160208101906104639190610863565b8585858181106104755761047561090d565b905060200201602081019061048a9190610923565b6105cf565b600101610430565b505050505050565b3360009081526001602052604090205460ff166104ef5760405162461bcd60e51b815260206004820152600e60248201526d1b9bdd081c9959da5cdd195c995960921b6044820152606401610221565b60008281526003602090815260408083203384529091529020541561054a5760405162461bcd60e51b8152602060048201526011602482015270185b1c9958591e481cdd589b5a5d1d1959607a1b6044820152606401610221565b600082815260036020908152604080832033808552908352928190208490555183815284917fad15c9b1f2faea53c216f9f3c9ba412efb7fe04068f82fef2112092ba287e840910160405180910390a35050565b6000546001600160a01b031633146105c85760405162461bcd60e51b8152600401610221906108ea565b6103c68383835b60008381526003602090815260408083206001600160a01b038616845290915290205461062a5760405162461bcd60e51b81526020600482015260096024820152686e6f2075706461746560b81b6044820152606401610221565b6001600160a01b03821660009081526002602052604090205461ffff16600a82610655576000610659565b6127105b610664906002610954565b61ffff1661067383600861097a565b61067d9190610997565b61068791906109aa565b6001600160a01b038416600081815260026020908152604091829020805461ffff191661ffff861617905590518515158152929350909186917f71d0e65986ef9b60389b87aa0fe902267f31655cf71ea529a6cc1f9bc970911a910160405180910390a350505050565b6000806040838503121561070457600080fd5b50508035926020909101359150565b60008083601f84011261072557600080fd5b50813567ffffffffffffffff81111561073d57600080fd5b6020830191508360208260051b850101111561075857600080fd5b9250929050565b6000806020838503121561077257600080fd5b823567ffffffffffffffff81111561078957600080fd5b61079585828601610713565b90969095509350505050565b6000806000806000606086880312156107b957600080fd5b85359450602086013567ffffffffffffffff808211156107d857600080fd5b6107e489838a01610713565b909650945060408801359150808211156107fd57600080fd5b5061080a88828901610713565b969995985093965092949392505050565b80356001600160a01b038116811461083257600080fd5b919050565b6000806040838503121561084a57600080fd5b8235915061085a6020840161081b565b90509250929050565b60006020828403121561087557600080fd5b61087e8261081b565b9392505050565b60006020828403121561089757600080fd5b5035919050565b8035801515811461083257600080fd5b6000806000606084860312156108c357600080fd5b833592506108d36020850161081b565b91506108e16040850161089e565b90509250925092565b6020808252600990820152683737ba1037bbb732b960b91b604082015260600190565b634e487b7160e01b600052603260045260246000fd5b60006020828403121561093557600080fd5b61087e8261089e565b634e487b7160e01b600052601160045260246000fd5b61ffff8181168382160280821691908281146109725761097261093e565b505092915050565b80820281158282048414176109915761099161093e565b92915050565b808201808211156109915761099161093e565b6000826109c757634e487b7160e01b600052601260045260246000fd5b50049056fea2646970667358221220d108b39ca11dfcb1a1aa35866ca5b29572a9bb43891fb249359a00ef6f8e2f5d64736f6c63430008180033'
SOURCE = r"""
// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

/// Minimal on-chain ledger for blockchain-based federated learning:
/// clients commit the hash of their update, an aggregator records accept/reject
/// decisions (which update an on-chain reputation), and the hash of the global model is committed.
contract FLLedger {
    address public owner;
    mapping(address => bool) public registered;
    mapping(address => uint16) public reputation;               // 0..10000 (basis points)
    mapping(uint256 => mapping(address => bytes32)) public updateHash;
    mapping(uint256 => bytes32) public globalHash;

    event UpdateSubmitted(uint256 indexed round, address indexed client, bytes32 hash);
    event Decision(uint256 indexed round, address indexed client, bool accepted);
    event GlobalCommitted(uint256 indexed round, bytes32 hash);

    constructor() { owner = msg.sender; }

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }

    function registerClients(address[] calldata clients) external onlyOwner {
        for (uint256 i = 0; i < clients.length; i++) {
            registered[clients[i]] = true;
            reputation[clients[i]] = 10000;
        }
    }

    function submitUpdate(uint256 round, bytes32 h) external {
        require(registered[msg.sender], "not registered");
        require(updateHash[round][msg.sender] == bytes32(0), "already submitted");
        updateHash[round][msg.sender] = h;
        emit UpdateSubmitted(round, msg.sender, h);
    }

    function _decide(uint256 round, address c, bool ok) internal {
        require(updateHash[round][c] != bytes32(0), "no update");
        uint256 r = reputation[c];
        r = (r * 8 + (ok ? 10000 : 0) * 2) / 10;                 // EMA with weight 0.2
        reputation[c] = uint16(r);
        emit Decision(round, c, ok);
    }

    function recordDecision(uint256 round, address c, bool ok) external onlyOwner {
        _decide(round, c, ok);
    }

    function recordDecisions(uint256 round, address[] calldata cs, bool[] calldata oks) external onlyOwner {
        require(cs.length == oks.length, "length");
        for (uint256 i = 0; i < cs.length; i++) _decide(round, cs[i], oks[i]);
    }

    function commitGlobal(uint256 round, bytes32 h) external onlyOwner {
        require(globalHash[round] == bytes32(0), "already committed");
        globalHash[round] = h;
        emit GlobalCommitted(round, h);
    }
}
"""


def run(n_clients, rounds):
    backend = PyEVMBackend(
        genesis_parameters=PyEVMBackend.generate_genesis_params(overrides={"gas_limit": 30_000_000}),
        genesis_state=PyEVMBackend.generate_genesis_state(num_accounts=n_clients + 1))
    w3 = Web3(EthereumTesterProvider(EthereumTester(backend)))
    owner, clients = w3.eth.accounts[0], w3.eth.accounts[1:n_clients + 1]
    w3.eth.default_account = owner

    def gas_of(tx_hash):
        return w3.eth.wait_for_transaction_receipt(tx_hash)["gasUsed"]

    C = w3.eth.contract(abi=ABI, bytecode=BYTECODE)
    rcpt = w3.eth.wait_for_transaction_receipt(C.constructor().transact())
    ct = w3.eth.contract(address=rcpt["contractAddress"], abi=ABI)
    g = {"deploy": rcpt["gasUsed"], "register_all": gas_of(ct.functions.registerClients(clients).transact())}

    submit, decide_single, decide_batch, commit = [], [], [], []
    for rnd in range(1, rounds + 1):
        for k, c in enumerate(clients):
            h = hashlib.sha256(f"update-{rnd}-{k}".encode()).digest()
            submit.append(gas_of(ct.functions.submitUpdate(rnd, h).transact({"from": c})))
        oks = [k % 5 != 0 for k in range(n_clients)]  # e.g. 20% of updates rejected
        if rnd == 1:
            for c, ok in zip(clients, oks):
                decide_single.append(gas_of(ct.functions.recordDecision(rnd, c, ok).transact()))
        else:
            decide_batch.append(gas_of(ct.functions.recordDecisions(rnd, clients, oks).transact()))
        gh = hashlib.sha256(f"global-{rnd}".encode()).digest()
        commit.append(gas_of(ct.functions.commitGlobal(rnd, gh).transact()))

    per_round = statistics.mean(submit) * n_clients + statistics.mean(decide_batch) + statistics.mean(commit)
    return {
        "clients": n_clients,
        "deploy_gas": g["deploy"],
        "register_all_gas": g["register_all"],
        "submit_update_gas_mean": round(statistics.mean(submit)),
        "decision_single_gas_mean": round(statistics.mean(decide_single)),
        "decisions_batch_gas_mean": round(statistics.mean(decide_batch)),
        "decisions_batch_gas_per_client": round(statistics.mean(decide_batch) / n_clients),
        "commit_global_gas_mean": round(statistics.mean(commit)),
        "total_gas_per_round": round(per_round),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", nargs="+", type=int, default=[10, 50, 100])
    ap.add_argument("--rounds", type=int, default=3, help="rounds measured (round 1 uses single decisions)")
    ap.add_argument("--out", default="gas_results.csv")
    a = ap.parse_args()
    assert a.rounds >= 2, "need at least 2 rounds"
    rows = []
    for n in a.clients:
        r = run(n, a.rounds)
        rows.append(r)
        print(r, flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("saved", a.out, "| compiler:", SOLC_VERSION)


if __name__ == "__main__":
    main()
