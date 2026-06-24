redef enum Notice::Type += { 
    Anomaly_Detected,
    Attack_Blocked,
    Suspicious_Activity,
    Connection_Denial,
    Malware_Detected,
    Botnet_Detected,
    SQL_Injection,
    Reconnaissance
};

global drop_connection: function(c: connection);

function drop_connection(c: connection)
{
    print fmt("[BLOCKING] Stopping traffic from: %s", c$id$orig_h);
    NOTICE([$note=Attack_Blocked, 
            $msg=fmt("Active Response Triggered for %s", c$id$orig_h),
            $conn=c]);
}