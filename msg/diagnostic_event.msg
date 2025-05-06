Header header

# The node that triggered or experienced the event
string origin_node

# Severity or importance of the event
# 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
uint8 event_severity

# A short textual description of the event
string event_description

# Additional key-value data about the event
key_value[] values