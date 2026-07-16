---- MODULE CounterService ----

CONSTANTS
    MaxValue

VARIABLES
    value

Init == value = 0

Increment ==
    /\ value < MaxValue
    /\ value' = value + 1

Get ==
    /\ value' = value

Next ==
    Increment \/ Get

TypeOK ==
    /\ value \in 0..MaxValue

NoLostUpdates ==
    \A x, y \in 0..MaxValue:
        /\ x < y
        => (value = x ~> value = y)
            => (\E k \in x..y-1: value = k ~> value = k+1)

Spec ==
    Init /\ [][Next]_<<value>>

THEOREM Spec => []TypeOK
THEOREM Spec => []NoLostUpdates

====