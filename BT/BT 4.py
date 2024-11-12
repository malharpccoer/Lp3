""" Write a program in solidirty to create student data use following constructs 
1. Structures
2. Arrays
3.Fallback
Deploy this as smart contract on Ethereum and Observe the transaction fee and gas values"""


SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Studendata{

    //Structures
    struct Student{
        string name;
        uint256 rollno;
    }

    //Array
    Student[] public studentarr;

    function addStudent(string memory name, uint rollno) public {
        for(uint i=0;i<studentarr.length;i++){
            if(studentarr[i].rollno==rollno){
                revert("Student already exists");
            }
        }
        studentarr.push(Student(name, rollno));
    }

    function getstudentLength() public view returns(uint){
        return studentarr.length;
    }

    function displayStudent() public view returns (Student[] memory){
        return studentarr;
    }

    function getByIndex(uint idx) public view returns(Student memory){
        require(idx< studentarr.length, "Index out of bound");
        return studentarr[idx];
    }

    //fallback
    fallback() external payable { 
        //This function handles external function call that doesnt exist
    }
    receive() external payable { 
        //This function will handle the ether sent by external user but without data mentioned
    }
}
