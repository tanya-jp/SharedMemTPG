#ifndef API_CLIENT_H
#define API_CLIENT_H

#include <string>
#include <vector>

class APIClient
{
public:
    APIClient(const std::string &apiToken,
              const std::string &experimentKey);

    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp);

    std::string MakeRequest(const std::string &url, const std::string &method,
                            const std::string &body,
                            const std::vector<std::string> &headers);

    void LogMetric(const std::string &metricName,
                   const std::string &metricValue);

private:
    std::string _baseUrl = "https://www.comet.com";
    std::string _apiToken;
    std::string _experimentKey;
};

#endif // API_CLIENT_H