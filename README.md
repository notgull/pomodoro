# pomodoro

A quick implementation of a pomodoro timer. This uses smol as an async runtime, breadx as a GUI interface,
cpal to produce a beeping noise at the end of the alarm, and rusttype to render the text.

# The Pomodoro Method

* Each work period is 25 minutes
* After every work period is a break period of 5 minutes
* Every 3rd break period is 20 minutes
