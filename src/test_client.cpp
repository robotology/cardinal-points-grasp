#include <yarp/os/all.h>
#include <stdio.h>

using namespace yarp::os;
using namespace std;

int main(int argc, char *argv[])
{
    if (argc<2) {
            yError() << "Supply targt port name as command-line argument 1";
            return 1;
        }

    Network yarp;
    string server_port_name = argv[1];
    string client_port_name = "/graspProcessor/test-client";
    int trials = 50;
    int successes = 0;

    RpcClient port;

    port.open(client_port_name);

    if (port.getOutputCount() == 0)
    {
        yInfo() <<  "connecting test client to " << server_port_name;
        yarp.connect(client_port_name, server_port_name);
    }

    for (int i = 0; i<trials && port.getOutputCount()!=0; i++)
    {
        yDebug() << "Performing request " << i;
        Bottle msg, rply;
        msg.addString("from_off_file");
        msg.addString("../../data/PC/Bottle_point_cloud_001.off");
        msg.addString("left");
        port.write(msg,rply);
        if (rply.toString() == "[ack]")
            successes++;
    }

    if (port.getOutputCount() == 0)
    {
        yError() << "Connection to server lost. Test failed";
        return 1;
    }


    yInfo() << "Test ended: " << successes << " out of " << trials << " trials.";











    return 0;
}
