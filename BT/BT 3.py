""" Write a smart contract on ates network for bank accunt for the following operations:
1.Deposit MOney
2.Withdraaw money
3.Show Balance"""

Method 1: 

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract bank{
    mapping(address => uint256) balances;

    function depositMoney(uint256 amount) public { 
        require(amount >= 0, "Amount must be greater than zero");
        balances[msg.sender] = balances[msg.sender] + amount;
    }

    function withdrawMoney(uint256 amount) public  { 
        require(amount <= balances[msg.sender], "Insufficient Balance");
        balances[msg.sender] = balances[msg.sender] - amount;
    }

    function showBalance() public view returns (uint256){ 
        return balances[msg.sender];
    }
}

Method 2:

// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Bank{
    uint256 balance=0;
    address public accOwner;

    constructor(){
        accOwner= msg.sender;
    }

    function Deposit() public  payable {
        require(accOwner==msg.sender, "Your are not owner");
        require(msg.value > 0, "Amount should be greater than 0!");
        balance += msg.value;
    }

    function Withdraw() public payable {
        require(accOwner==msg.sender, "Your are not owner");
        require(msg.value > 0, "Amount should be greater than 0!");
        balance -= msg.value;
    }

    function showBalance() public  view returns(uint256){
        require(accOwner==msg.sender, "You are not owner");
        return balance;
    }
}
