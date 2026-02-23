import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import DatasetPage from './pages/DatasetPage';
import ExperimentPage from './pages/ExperimentPage';
import HistoryPage from './pages/HistoryPage';
import ComparePage from './pages/ComparePage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/datasets" element={<DatasetPage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
