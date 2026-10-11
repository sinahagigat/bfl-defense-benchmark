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
