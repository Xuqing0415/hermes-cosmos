---- MODULE GenericService ----

DESCRIPTION
    Generated from natural language requirement: 实现 http 计数器服务，两个操作：post /inc 增加计数；get /get 返回当前值。要...

CONSTANTS
    None

VARIABLES
    state

Init == state = "initial"

Operations ==
    read

Next ==
    TRUE

Spec ==
    Init /\ [][Next]_<<state>>

====
