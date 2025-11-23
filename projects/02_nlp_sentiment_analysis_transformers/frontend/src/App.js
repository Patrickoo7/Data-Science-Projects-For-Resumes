import React, { useState, useEffect } from 'react';
import {
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Button,
  Card,
  CardContent,
  TextField,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Paper,
  Tab,
  Tabs
} from '@mui/material';
import {
  SentimentVeryDissatisfied,
  SentimentNeutral,
  SentimentVerySatisfied,
  Analytics,
  Key
} from '@mui/icons-material';
import axios from 'axios';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

function SentimentIcon({ sentiment }) {
  if (sentiment === 'positive') {
    return <SentimentVerySatisfied sx={{ color: 'success.main', fontSize: 40 }} />;
  } else if (sentiment === 'negative') {
    return <SentimentVeryDissatisfied sx={{ color: 'error.main', fontSize: 40 }} />;
  } else {
    return <SentimentNeutral sx={{ color: 'warning.main', fontSize: 40 }} />;
  }
}

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '');
  const [tab, setTab] = useState(0);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (apiKey) {
      fetchStats();
    }
  }, [apiKey]);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/analytics/usage/summary`, {
        headers: { 'X-API-Key': apiKey }
      });
      setStats(response.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if (!apiKey) {
      setError('Please enter your API key');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${API_URL}/sentiment/predict`,
        { text },
        {
          headers: {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
          }
        }
      );

      setResult(response.data);
      localStorage.setItem('apiKey', apiKey);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🤖 SentiAI - Sentiment Analysis Platform
            </Typography>
            <Typography variant="body2">
              v1.0.0
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Tabs value={tab} onChange={(e, v) => setTab(v)} sx={{ mb: 3 }}>
            <Tab label="Analyze" />
            <Tab label="Analytics" icon={<Analytics />} iconPosition="start" />
            <Tab label="API Keys" icon={<Key />} iconPosition="start" />
          </Tabs>

          {tab === 0 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Analyze Sentiment
                    </Typography>

                    <TextField
                      fullWidth
                      label="API Key"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      margin="normal"
                      helperText="Enter your API key to get started"
                    />

                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      label="Enter text to analyze"
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      margin="normal"
                      placeholder="This product is amazing! Best purchase ever."
                    />

                    <Button
                      variant="contained"
                      size="large"
                      onClick={analyzeSentiment}
                      disabled={loading}
                      sx={{ mt: 2 }}
                      fullWidth
                    >
                      {loading ? <CircularProgress size={24} /> : 'Analyze Sentiment'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>

              {error && (
                <Grid item xs={12}>
                  <Alert severity="error">{error}</Alert>
                </Grid>
              )}

              {result && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h5" gutterBottom>
                        Results
                      </Typography>

                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <SentimentIcon sentiment={result.sentiment} />
                        <Box sx={{ ml: 2 }}>
                          <Typography variant="h4">
                            {result.sentiment.toUpperCase()}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Confidence: {(result.confidence * 100).toFixed(2)}%
                          </Typography>
                        </Box>
                      </Box>

                      <Typography variant="subtitle1" gutterBottom>
                        Probabilities:
                      </Typography>

                      <Grid container spacing={2}>
                        <Grid item xs={4}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light' }}>
                            <Typography variant="h6">Negative</Typography>
                            <Typography variant="h4">
                              {(result.probabilities.negative * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={4}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                            <Typography variant="h6">Neutral</Typography>
                            <Typography variant="h4">
                              {(result.probabilities.neutral * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={4}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                            <Typography variant="h6">Positive</Typography>
                            <Typography variant="h4">
                              {(result.probabilities.positive * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>

                      <Box sx={{ mt: 2 }}>
                        <Chip
                          label={`Processed in ${result.processing_time_ms}ms`}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Chip
                          label={`Model: ${result.model_version}`}
                          size="small"
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          )}

          {tab === 1 && stats && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Total Predictions
                    </Typography>
                    <Typography variant="h4">
                      {stats.total_predictions.toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Avg Confidence
                    </Typography>
                    <Typography variant="h4">
                      {(stats.average_confidence * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Sentiment Breakdown
                    </Typography>
                    <Grid container spacing={1}>
                      {Object.entries(stats.sentiment_breakdown || {}).map(([key, value]) => (
                        <Grid item xs={4} key={key}>
                          <Chip
                            label={`${key}: ${value}`}
                            color={
                              key === 'positive' ? 'success' :
                              key === 'negative' ? 'error' : 'warning'
                            }
                          />
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {tab === 2 && (
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  API Key Management
                </Typography>
                <Alert severity="info">
                  To get your API key, please register at the API endpoint:
                  POST {API_URL}/auth/register
                </Alert>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Current API Key: {apiKey ? '••••••••' : 'Not set'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </Container>

        <Box
          component="footer"
          sx={{
            py: 3,
            px: 2,
            mt: 'auto',
            backgroundColor: (theme) =>
              theme.palette.mode === 'light'
                ? theme.palette.grey[200]
                : theme.palette.grey[800],
          }}
        >
          <Container maxWidth="sm">
            <Typography variant="body2" color="text.secondary" align="center">
              © 2024 SentiAI Platform. Powered by BERT & Transformers.
            </Typography>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
