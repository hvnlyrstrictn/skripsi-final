event http_request(c: connection, method: string, original_URI: string, unescaped_URI: string, version: string)
{
    if ( c$id$orig_h == 192.168.1.3 )
    {
        NOTICE([
            $note = Suspicious_Activity,
            $msg  = fmt("Suspicious HTTP from %s", c$id$orig_h)
        ]);

        drop_connection(c);
    }
}

