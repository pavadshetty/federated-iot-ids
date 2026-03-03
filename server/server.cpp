#include <iostream>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

#define PORT 9000
#define CLIENTS 2   // change to 5 if using 5 RPis

/* -------- Safe Receive -------- */
bool recv_all(int sock, void* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        ssize_t bytes = recv(sock, (char*)buffer + total, length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}

/* -------- Safe Send -------- */
bool send_all(int sock, const void* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        ssize_t bytes = send(sock, (char*)buffer + total, length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}

int main() {

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(PORT);
    address.sin_addr.s_addr = INADDR_ANY;

    bind(server_fd, (sockaddr*)&address, sizeof(address));
    listen(server_fd, 5);

    cout << "🚀 Federated Server started on port " << PORT << endl;

    while (true) {

        cout << "\nWaiting for clients...\n";

        vector<int> client_sockets;

        // Accept clients
        for (int i = 0; i < CLIENTS; i++) {
            int client_sock = accept(server_fd, nullptr, nullptr);
            cout << "Client " << i + 1 << " connected\n";
            client_sockets.push_back(client_sock);
        }

        vector<vector<double>> client_weights(CLIENTS);
        int model_size;

        // Receive weights
        for (int i = 0; i < CLIENTS; i++) {

            if (!recv_all(client_sockets[i], &model_size, sizeof(int))) {
                cout << "Error receiving model size\n";
                return 0;
            }

            client_weights[i].resize(model_size);

            if (!recv_all(client_sockets[i],
                          client_weights[i].data(),
                          model_size * sizeof(double))) {
                cout << "Error receiving weights\n";
                return 0;
            }

            cout << "Received weights from Client " << i + 1 << endl;
        }

        // Federated Averaging
        vector<double> global_weights(model_size, 0);

        for (int i = 0; i < CLIENTS; i++)
            for (int j = 0; j < model_size; j++)
                global_weights[j] += client_weights[i][j];

        for (int j = 0; j < model_size; j++)
            global_weights[j] /= CLIENTS;

        cout << "Global model averaged\n";

        // Send global model back
        for (int i = 0; i < CLIENTS; i++) {
            send_all(client_sockets[i],
                     global_weights.data(),
                     model_size * sizeof(double));
            close(client_sockets[i]);
        }

        cout << "Global model sent to clients\n";
    }

    close(server_fd);
    return 0;
}