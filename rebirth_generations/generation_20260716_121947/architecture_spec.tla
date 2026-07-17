---- MODULE AutoTestGenArchitecture ----
EXTENDS Naturals, TLC, Sequences

CONSTANTS ModuleNames, InterfaceNames

Modules == ModuleNames
Interfaces == InterfaceNames

(* 模块状态 *)
State == [
    modules: Modules -> {idle, working, error},
    data_flows: (Modules \X Modules) -> BOOLEAN,
    pending_tasks: Seq(InterfaceNames)
]

(* 初始状态 *)
Init == /\\
    \A m \in Modules: modules[m] = idle
    /\\
    \A m1, m2 \in Modules: data_flows[m1, m2] = FALSE
    /\\
    pending_tasks = <<>>

(* 启动模块 *)
StartModule(m) == /\\
    modules[m] = idle
    /\\
    modules' = [modules EXCEPT ![m] = working]

(* 停止模块 *)
StopModule(m) == /\\
    modules[m] = working
    /\\
    modules' = [modules EXCEPT ![m] = idle]

(* 数据流动 *)
DataFlow(m1, m2) == /\\
    modules[m1] = working
    /\\
    modules[m2] = working
    /\\
    data_flows[m1, m2] = FALSE
    /\\
    data_flows' = [data_flows EXCEPT ![m1, m2] = TRUE]

(* 完成数据流动 *)
CompleteFlow(m1, m2) == /\\
    data_flows[m1, m2] = TRUE
    /\\
    data_flows' = [data_flows EXCEPT ![m1, m2] = FALSE]

(* 添加任务 *)
AddTask(task) == /\\
    pending_tasks' = Append(pending_tasks, task)

(* 处理任务 *)
ProcessTask == /\\
    Len(pending_tasks) > 0
    /\\
    pending_tasks' = Tail(pending_tasks)

Next ==
    \E m \in Modules: StartModule(m)
    \/ \E m \in Modules: StopModule(m)
    \/ \E m1, m2 \in Modules: DataFlow(m1, m2)
    \/ \E m1, m2 \in Modules: CompleteFlow(m1, m2)
    \/ AddTask("")
    \/ ProcessTask

(* 安全性属性：无死锁 *)
NoDeadlock ==
    ~(modules = [m \in Modules |-> working] /\\ data_flows = [m1, m2 \in Modules |-> FALSE])

(* 活性属性：所有任务都会被处理 *)
AllTasksProcessed ==
    [] (Len(pending_tasks) > 0 => <><<ProcessTask>>>)

(* 模块可用性 *)
ModuleAvailable(m) ==
    [] (<>modules[m] = working)

====
