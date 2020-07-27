# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:01:11 2020

@author: 佘建友
"""

'------------------------------------------------------------------------------'
'------------------------------------------------------------------------------'
"""
容器为常用数据结构的一类。
容器的重要特征是：元素(element)，与之对应的索引(index)。
容器支持对这些元素的保存、访问、支持、删除。

根据元素存入和取出的顺序，分为：栈和队列

Such feature of stack is known as Last in First Out(LIFO) feature.
The operations of adding and removing the elements is known as PUSH and POP.

"""
'------------------------------------------------------------------------------'
class Stack(object):
    
    def __init__(self):
        self.stack = []
        
    def add(self,dataval):
        """
        Use list append method to add element(PUSH)
        """
        if dataval not in self.stack:
            self.stack.append(dataval)#Add an item to the end of the list
            return True
        else:
            print("the element has exited")
    
    def peak(self):
        """
        Use peek to look at the top of the stack
        """
        return self.stack[-1]
    
    def remove(self):
        """
        Use list pop method to remove element(POP)
        """
        if len(self.stack) <= 0:
            return ("No element in ths Stack")
        else:
            return self.stack.pop()#If no index is specified, a.pop() removes and returns the last item in the list. 

'------------------------------------------------------------------------------'
"""
Such feature of Queue is known as Last in First-in-First feature.

"""  
class Queue(object):
    
    def __init__(self):
        self.queue = list()
        
    def addtop(self,dataval):
        """
        Insert method to add element
        """
        if dataval not in self.queue:
            self.queue.insert(0,dataval)#Insert an item at a given position.
            return True
        return False
    
    def size(self):
        """
        the number of elements in queue
        """
        return len(self.queue)
    
    def removefromq(self):
        """
        Pop method to remove element
        """
        if len(self.queue) > 0:
            return self.queue.pop()
        return ("No elements in Queue!")
    
'------------------------------------------------------------------------------'    
''
"""
A double-ended queue,or deque,supports adding and removing elementd from either
end.
The more commonly used stacks and queues are degenerate forms of deques,
where the inputs and outputs are restricted a single end.
"""

class Dequeue(object):
    
    def __init__(self):
        self.dequeue = list()
        
    def addright(self,dataval):
        """
        Insert method to add element in top
        """
        if dataval not in self.dequeue:
            self.dequeue.insert(0,dataval)
            return True
        return False
    
    def addleft(self,dataval):
        """
        Insert method to add element in bottom
        """
        if dataval not in self.dequeue:
            self.dequeue.insert(len(self.dequeue),dataval)
            return True
        return False
    
    def removeleft(self):
        """
        Pop method to remove element in top
        """
        if len(self.dequeue) > 0:
            return self.dequeue.pop()
        return ("No elements in Dequeue!")
    
    def removeright(self):
        """
        Pop method to remove element in bottom
        """
        if len(self.dequeue) > 0:
            return self.dequeue.remove()#Remove the first item from the list 
        return ("No elements in Dequeue!")


'-----------------------------------------------------------------------------'
'-----------------------------------------------------------------------------'       
"""
线性表：是一些元素的序列，即元素的线性关系。
线性表的基本需要是：
                1、能够找到表中的首元素；
                2、从表里的任一元素出发，可以找到它之后的下一个元素；
采用链接方式实现线性表的基本想法：
                1、把表中的元素分别存储在一批独立的存储空间里；
                2、保证从组成表结构中的任一个结点可找到与其相关的下一个结点(node)
                3、在前一结点采用链接的方式显式地记录与下一结点之间关联
下面各种类型的链接表只考虑：
                每个结点李只存储一个表元素   
"""

'------------------------------------------------------------------------------'
"""
node = (elem,link)
A linked list is a sequence of data elemets,which are connected together via links.
It means each nodes contains a connection to another data element in form of 
a pointer,which is impelemented by defining a node class,where we pass the appropriate
the values thorugh the node object to point to the next data elements.
"""
class Node():
    
    def __init__(self,dataval=None):
        """
        the fuction:
            passing the values thorugh the node object to point to 
            the next data elements.
        """
        self.dataval = dataval
        self.nextval = None
        
class SLinkedList():
    
    def __init__(self):
        self.headval = None#表头
        
    """
    the ways:
        list1 = SLinkedList()
        list1.headval = Node("佘建友")
        e2 = Node("杨丽蓉")
        e3 = Node("小佘建友")
        
        Link first Node to second node
        list1.headval.nextval = e2
    
        Link second Node to thied node
        e2.nextval=e3
    """
    
    def listprint(self):
        """
        function:
                Print the linked list
        -----------------------------------------------------------------------
        note:
        Singly linked list can be traversed in only forward direction starting from the 
        first data element.
        
        printing the value of the next data element by assgining the pointer of the next node
        to the current data element.
        """
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval
            
    
    """
    Inserting element in the linked list involves reassigning the pointers from 
    the existing nodes to newly inserted node.
    
    Depending on whether the new data element is getting inserted at the beginning
    or at the middled or at the end of the linked list,the way has variable kinds:
    """
    
    """
    Inserting at the Beginning of the Linked List
    ---------------------------------------------------------------------------
    This involves pointing the next pointer of the new data node to the current
    head of the linked list.
    """
    def AtBegining(self,newdata):
        NewNode = Node(newdata)
        
        #Update the new nodes next val(the next pointer) to existing node
        NewNode.nextval = self.headval
        self.headval = NewNode
         
    """
    Inserting at the End of the Linked List
    ---------------------------------------------------------------------------
    This involves pointing the next pointer of the current node to the new data
    node in the linked list
    """
    def AtEnd(self,newdata):
        """
        Function:
            to add newnode
        """
        NewNode = Node(newdata)
        if self.headval is None:
            self.headval = NewNode
            return 
        laste = self.headval
        while (laste.nextval):
            laste = laste.nextval
        laste.nextval=NewNode
        
    """
    Inserting in between two Data Nodes
    ---------------------------------------------------------------------------
    This involves changing the pointer of a specific node to the new node.
    """
    def Inbetween(self,middle_node,newdata):
        """
        Function:
            to add newnode
        -----------------------------------
        Params:
            middle_node:the pointer of the last node
        """
        if middle_node is None:
            print("The mentioned node is absent")
            return
        
        NewNode = Node(newdata)
        NewNode.nextval = middle_node.nextval
        middle_node.nextval = NewNode
        
        
    """
    Removing an item from a linked list
    ---------------------------------------------------------------------------
    We can remove an existing node using the key for that node.
    Locate the previos node of the node which is to be deleted.
    Then point the next poniter of this node to the next node of the node to be deleted.
    Then the previos pointer of the node point to the next node of it.
    """
    def RemoveNode(self,Removekey):
        HeadVal = self.headval#表头
        
        if (HeadVal is not None):
            if (HeadVal.dataval == Removekey):#Locate
                self.headval = HeadVal.nextval#point the next pointer of this node
                HeadVal = None#del the current node
                return
            
        while (HeadVal is not None):#while looop is the locator
            if HeadVal.dataval == Removekey:#Locate
                break
            
            prev = HeadVal#Locate the last node
            HeadVal = HeadVal.nextval#look for the current node
            
        if (HeadVal == None):
            return
        
        prev.nextval = HeadVal.nextval#del the current node by changing nextval of the previous node 
        #to the node of the current node
        
        HeadVal = None
        
    

    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
























    
        
