# Super Pong
The super-pong server is a C program with an Internet domain stream socket to receive connections from the various clients,
up to a maximum of ten simultaneously.

Whenever the server receives one message, it should update the state of the game and send suitable messages representing the
updated state to all active clients.

The server should store an array of all the clients with all the relevant information (e.g. paddle position). A client is
inserted into this list when such client connects and removed when it disconnects.

The client should connect to a super-pong server for the user to play. The address of the server should be supplied to the
client as a command line argument. After connecting, the client will start reading the keyboard. If the user presses one of
the arrow keys, a message should be sent to the server.

Whenever the client receives an message, the client should draw the ball and all the paddles.
